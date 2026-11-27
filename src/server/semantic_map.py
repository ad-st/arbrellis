#!/usr/bin/env python3
"""Console prototype of a personal semantic citation map.

Each selected passage becomes one dot. Semantic neighbours make that dot larger,
while a short extractive label is chosen from phrases that actually occur in
the selected passage.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding


MODEL = "BAAI/bge-small-en-v1.5"
CACHE = Path.home() / ".cache" / "arbrellis" / "models"

# These words are useful for sentence structure but make poor dot labels.
STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "although", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "between", "both",
    "but", "by", "can", "could", "did", "do", "does", "doing", "during", "each",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "may", "might", "more", "most", "no",
    "nor", "not", "now", "of", "on", "once", "only", "or", "other", "our", "out",
    "over", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "use", "used", "uses", "using", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "why",
    "will", "with", "would", "you", "your",
}

# A tiny, deliberately transparent substitute for a POS tagger. These common
# verbs delimit title-like noun phrases in the demo. A production extractor
# should replace this list with proper part-of-speech tagging or a title model.
LABEL_BREAKS = (STOPWORDS - {"of", "for", "in"}) | {
    "allow", "allows", "control", "controls", "give", "gives", "learn", "learns",
    "let", "lets", "matter", "matters", "observe", "observes", "observing",
    "prevent", "prevents", "reduce", "reduces", "reducing", "retain", "retains",
    "retaining", "speed", "speeds", "speeding", "store", "stores",
}

DEMO_PASSAGES = [
    "Attention lets a language model give more weight to the tokens that matter for the current prediction.",
    "Multi-head attention learns several relevance patterns between words at the same time.",
    "Quantization stores neural-network weights with fewer bits, reducing memory use and often speeding up inference.",
    "An eight-bit quantized model is smaller than its full-precision version while usually retaining similar quality.",
    "Database transactions use isolation levels to control which concurrent changes are visible.",
    "Serializable isolation prevents transactions from observing inconsistent concurrent database states.",
    "Attention layers can also be quantized, although aggressive low-bit conversion may reduce model accuracy.",
]


def normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


class Encoder:
    def __init__(self, model_name: str = MODEL) -> None:
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(CACHE), threads=4)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = np.asarray(list(self.model.embed(list(texts), batch_size=64)), dtype=np.float32)
        return normalize(values)


@dataclass
class Dot:
    vector: np.ndarray
    citation: str
    label: str = "untitled"


def candidate_phrases(texts: list[str]) -> list[str]:
    """Extract grounded 1-5 word phrases without a linguistic model."""
    candidates: dict[str, str] = {}
    for text in texts:
        words = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*", text)
        lower = [word.lower() for word in words]
        for start in range(len(words)):
            if lower[start] in STOPWORDS:
                continue
            for length in range(1, 6):
                end = start + length
                if end > len(words):
                    break
                phrase_words = lower[start:end]
                if phrase_words[-1] in STOPWORDS:
                    continue
                if any(word in LABEL_BREAKS for word in phrase_words):
                    continue
                # Avoid candidates dominated by glue words.
                content = sum(word not in STOPWORDS for word in phrase_words)
                if content < max(1, math.ceil(length / 2)):
                    continue
                key = " ".join(phrase_words)
                candidates.setdefault(key, " ".join(words[start:end]))
    # Limit inference work while preferring phrases repeated across citations.
    ranked = sorted(
        candidates,
        key=lambda phrase: (
            -sum(text.lower().count(phrase) for text in texts),
            abs(len(phrase.split()) - 3),
            phrase,
        ),
    )
    return [candidates[key] for key in ranked[:160]]


def choose_label(citation: str, vector: np.ndarray, encoder: Encoder) -> str:
    phrases = candidate_phrases([citation])
    if not phrases:
        return "untitled"
    vectors = encoder.encode(phrases)
    semantic = vectors @ vector
    scores: list[float] = []
    for phrase, similarity in zip(phrases, semantic, strict=True):
        word_count = len(phrase.split())
        # BGE naturally favours longer fragments. Counterbalance that so these
        # labels act like titles rather than clipped sentences.
        length_penalty = 0.035 * max(0, word_count - 3) + 0.008 * abs(word_count - 2)
        scores.append(float(similarity) - length_penalty)
    return phrases[int(np.argmax(scores))].lower()


class SemanticMap:
    def __init__(self, encoder: Encoder, similarity_threshold: float) -> None:
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.dots: list[Dot] = []

    def add(self, passage: str) -> tuple[int, int, float | None]:
        vector = self.encoder.encode([passage])[0]
        label = choose_label(passage, vector, self.encoder)
        nearest_index, similarity = -1, None
        if self.dots:
            similarities = np.asarray([vector @ dot.vector for dot in self.dots])
            nearest_index = int(np.argmax(similarities))
            similarity = float(similarities[nearest_index])
        self.dots.append(Dot(vector, passage, label))
        return len(self.dots) - 1, nearest_index, similarity

    def similarities(self) -> np.ndarray:
        if not self.dots:
            return np.empty((0, 0), dtype=np.float32)
        matrix = np.asarray([dot.vector for dot in self.dots])
        return matrix @ matrix.T

    def coordinates(self) -> np.ndarray:
        """Project dot centroids to two dimensions with dependency-free PCA."""
        if not self.dots:
            return np.empty((0, 2), dtype=np.float32)
        matrix = np.asarray([dot.vector for dot in self.dots])
        centered = matrix - matrix.mean(axis=0)
        if len(self.dots) == 1:
            return np.zeros((1, 2), dtype=np.float32)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ axes[: min(2, len(axes))].T
        if projected.shape[1] == 1:
            projected = np.column_stack((projected, np.zeros(len(projected))))
        scale = np.max(np.abs(projected), axis=0)
        return 10 * projected / np.maximum(scale, 1e-9)

    def print_map(self) -> None:
        similarities = self.similarities()
        print("\nSEMANTIC DOTS")
        print("-" * 78)
        for index, (dot, position) in enumerate(zip(self.dots, self.coordinates(), strict=True), 1):
            neighbours = int(np.sum(similarities[index - 1] >= self.similarity_threshold)) - 1
            radius = 1 + math.log2(neighbours + 2)
            print(
                f"{index:>2}. ({position[0]:>6.2f}, {position[1]:>6.2f}) "
                f"size={radius:.1f}  similar={neighbours:>2}  {dot.label}"
            )
        print("\nSIMILARITY EDGES")
        print("-" * 78)
        found = False
        for left in range(len(self.dots)):
            for right in range(left + 1, len(self.dots)):
                score = float(similarities[left, right])
                if score >= self.similarity_threshold:
                    found = True
                    print(f"{left + 1:>2} <--> {right + 1:<2}  similarity={score:.3f}")
        if not found:
            print("(none at the current threshold)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use bundled example selections")
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.73,
        help="Minimum cosine similarity for linking two dots (default: 0.73)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {MODEL} locally...")
    semantic_map = SemanticMap(Encoder(), args.similarity)
    passages = DEMO_PASSAGES if args.demo else None

    if passages is None:
        print("Paste one selected passage per line. Submit an empty line to finish.\n")
        passages = []
        while passage := input("> ").strip():
            passages.append(passage)

    for number, passage in enumerate(passages, 1):
        index, nearest, similarity = semantic_map.add(passage)
        score = "first dot" if similarity is None else f"nearest dot {nearest + 1}, similarity {similarity:.3f}"
        print(f"[{number}] created dot {index + 1}: {semantic_map.dots[index].label!r} ({score})")

    semantic_map.print_map()
    print("\nOpen a dot by number:")
    for index, dot in enumerate(semantic_map.dots, 1):
        print(f"\n[{index}] {dot.label}")
        print(f"  - {dot.citation}")


if __name__ == "__main__":
    main()
