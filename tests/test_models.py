import os
import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "server"))

import local_knowledge_models  # noqa: E402
from local_knowledge_models import (  # noqa: E402
    QWEN_MODEL,
    QWEN_REVISION,
    LocalKnowledgeModels,
    normalize_label,
)


class ModelConfigurationTests(unittest.TestCase):
    def test_t5gemma_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(LocalKnowledgeModels().t5_enabled)

    def test_t5gemma_can_be_enabled(self):
        with patch.dict(os.environ, {"ARBRELLIS_ENABLE_T5GEMMA": "true"}, clear=True):
            self.assertTrue(LocalKnowledgeModels().t5_enabled)

    def test_label_normalization_removes_conversational_prefix(self):
        self.assertEqual(normalize_label("The label is: Speculative decoding"), "speculative decoding")

    def test_cache_status_does_not_trigger_model_loading(self):
        with TemporaryDirectory() as directory, patch.object(
            local_knowledge_models, "CACHE", Path(directory)
        ):
            models = LocalKnowledgeModels(t5_enabled=False)
            status = models.cache_status()
            self.assertFalse(status["ready"])
            self.assertIsNone(models._qwen_model)
            self.assertIsNone(models._encoder)

    def test_complete_qwen_snapshot_is_detected(self):
        with TemporaryDirectory() as directory, patch.object(
            local_knowledge_models, "CACHE", Path(directory)
        ):
            snapshot = LocalKnowledgeModels._snapshot_path(QWEN_MODEL, QWEN_REVISION)
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            status = LocalKnowledgeModels(t5_enabled=False).cache_status()
            self.assertTrue(status["models"]["qwen"]["cached"])


if __name__ == "__main__":
    unittest.main()
