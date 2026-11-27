#!/usr/bin/env python3
"""Local label generation and semantic ranking for knowledge capture."""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from semantic_map import CACHE, Encoder


T5_MODEL = "google/t5gemma-b-b-prefixlm-it"
QWEN_MODEL = "Qwen/Qwen3-0.6B"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
T5_REVISION = "f0debd90994f48ce76b61fff03330b353c00eafc"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CAPTION_VERSION = 1
ProgressCallback = Callable[[str, str], None]


def label_prompt(text: str) -> str:
    return (
        f'"{text}"\n'
        "Use 3 words to answer following question: what is the closest concept "
        "matching this sentence? Output only the answer."
    )


def caption_prompt(text: str) -> str:
    return (
        f"<FACT>\n{text}\n</FACT>\n\n"
        "Summarize the FACT in one short sentence.\n"
        "Write the statement directly, without introductory phrases.\n"
        "Output only the sentence."
    )


def normalize_label(value: str) -> str:
    """Turn occasional conversational output into a stable graph label."""
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    first_line = re.sub(
        r"^(?:the\s+)?(?:answer|concept|label)\s+is\s*:?\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", first_line)
    return " ".join(words).lower() or "untitled"


@dataclass
class KnowledgeAnalysis:
    label: str
    vector: np.ndarray
    candidates: list[dict[str, object]]
    agreement: bool
    timings_ms: dict[str, int]


class LocalKnowledgeModels:
    """Lazily load local models and serialize their inference."""

    def __init__(self, t5_enabled: bool | None = None) -> None:
        self._lock = threading.RLock()
        if t5_enabled is None:
            t5_enabled = os.environ.get("ARBRELLIS_ENABLE_T5GEMMA", "").lower() in {
                "1", "true", "yes", "on"
            }
        self.t5_enabled = t5_enabled
        self.t5_error: str | None = None
        self._t5_tokenizer = None
        self._t5_model = None
        self._qwen_tokenizer = None
        self._qwen_model = None
        self._encoder: Encoder | None = None

    def _load_encoder(self) -> None:
        if self._encoder is None:
            self._encoder = Encoder(BGE_MODEL)

    def _load_qwen(self) -> None:
        if self._qwen_model is not None:
            return
        cache = CACHE / "huggingface"
        torch.set_num_threads(4)
        print("Loading Qwen for local concept generation...", flush=True)
        self._qwen_tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, cache_dir=cache
        )
        self._qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, cache_dir=cache, dtype=torch.float32
        ).eval()

    def _load_t5(self) -> bool:
        if not self.t5_enabled or self.t5_error:
            return False
        if self._t5_model is not None:
            return True
        try:
            cache = CACHE / "huggingface"
            print("Loading optional T5Gemma model...", flush=True)
            self._t5_tokenizer = AutoTokenizer.from_pretrained(
                T5_MODEL, revision=T5_REVISION, cache_dir=cache
            )
            self._t5_model = AutoModelForSeq2SeqLM.from_pretrained(
                T5_MODEL, revision=T5_REVISION, cache_dir=cache, dtype=torch.float32
            ).eval()
            return True
        except Exception as error:
            self.t5_error = str(error)
            print(f"T5Gemma unavailable; continuing with Qwen: {error}", flush=True)
            return False

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            self._load_encoder()
            assert self._encoder is not None
            return self._encoder.encode(texts)

    def _t5_label(self, prompt: str) -> str:
        assert self._t5_tokenizer is not None and self._t5_model is not None
        inputs = self._t5_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self._t5_model.generate(
                **inputs, max_new_tokens=12, do_sample=False
            )
        return self._t5_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def caption_fact(self, text: str) -> str:
        """Create the secondary caption shown only inside expanded concepts."""
        with self._lock:
            if not self._load_t5():
                self._load_qwen()
                return self._qwen_generate(caption_prompt(text), max_new_tokens=32)
            inputs = self._t5_tokenizer.apply_chat_template(
                [{"role": "user", "content": caption_prompt(text)}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = self._t5_model.generate(
                    **inputs, max_new_tokens=32, do_sample=False
                )
            return self._t5_tokenizer.decode(
                output[0], skip_special_tokens=True
            ).strip()

    def _qwen_generate(self, prompt: str, max_new_tokens: int = 12) -> str:
        assert self._qwen_tokenizer is not None and self._qwen_model is not None
        formatted = self._qwen_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._qwen_tokenizer(formatted, return_tensors="pt")
        with torch.inference_mode():
            output = self._qwen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._qwen_tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        return self._qwen_tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True
        ).strip()

    def analyze(
        self, text: str, progress: ProgressCallback | None = None
    ) -> KnowledgeAnalysis:
        with self._lock:
            prompt = label_prompt(text)

            t5_raw = None
            t5_ms = 0
            if self.t5_enabled:
                if progress:
                    progress("t5gemma", "running")
                started = time.perf_counter()
                if self._load_t5():
                    t5_raw = self._t5_label(prompt)
                    t5_ms = round((time.perf_counter() - started) * 1000)
                if progress:
                    progress("t5gemma", "complete" if t5_raw else "disabled")
            elif progress:
                progress("t5gemma", "disabled")

            if progress:
                progress("qwen", "running")
            started = time.perf_counter()
            self._load_qwen()
            qwen_raw = self._qwen_generate(prompt)
            qwen_ms = round((time.perf_counter() - started) * 1000)
            if progress:
                progress("qwen", "complete")

            qwen_label = normalize_label(qwen_raw)
            generated = [("qwen", qwen_raw, qwen_label)]
            if t5_raw is not None:
                generated.insert(0, ("t5gemma", t5_raw, normalize_label(t5_raw)))
            unique_labels = list(dict.fromkeys(item[2] for item in generated))
            if progress:
                progress("bge", "running")
            started = time.perf_counter()
            vectors = self.embed([text, *unique_labels])
            bge_ms = round((time.perf_counter() - started) * 1000)
            if progress:
                progress("bge", "complete")
            scores = {
                label: float(vectors[0] @ vectors[index + 1])
                for index, label in enumerate(unique_labels)
            }
            agreement = len(unique_labels) == 1
            selected = unique_labels[0] if agreement else max(unique_labels, key=scores.get)
            candidates = [
                {
                    "model": model,
                    "raw": raw,
                    "label": label,
                    "similarity": round(scores[label], 4),
                    "selected": selected == label,
                }
                for model, raw, label in generated
            ]
            timings = {"qwen": qwen_ms, "bge_large": bge_ms}
            if t5_raw is not None:
                timings["t5gemma"] = t5_ms
            return KnowledgeAnalysis(
                label=selected,
                vector=vectors[0].astype(np.float32),
                candidates=candidates,
                agreement=agreement,
                timings_ms=timings,
            )

    def embed_with_label(
        self,
        text: str,
        label: str,
        progress: ProgressCallback | None = None,
    ) -> KnowledgeAnalysis:
        """Refresh semantic evidence while preserving an already assigned label."""
        if progress:
            progress("bge", "running")
        started = time.perf_counter()
        vector = self.embed([text])[0].astype(np.float32)
        bge_ms = round((time.perf_counter() - started) * 1000)
        if progress:
            progress("bge", "complete")
        return KnowledgeAnalysis(
            label=label,
            vector=vector,
            candidates=[],
            agreement=True,
            timings_ms={"bge_large": bge_ms},
        )
