import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "server"))

from local_knowledge_models import LocalKnowledgeModels, normalize_label  # noqa: E402


class ModelConfigurationTests(unittest.TestCase):
    def test_t5gemma_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(LocalKnowledgeModels().t5_enabled)

    def test_t5gemma_can_be_enabled(self):
        with patch.dict(os.environ, {"ARBRELLIS_ENABLE_T5GEMMA": "true"}, clear=True):
            self.assertTrue(LocalKnowledgeModels().t5_enabled)

    def test_label_normalization_removes_conversational_prefix(self):
        self.assertEqual(normalize_label("The label is: Speculative decoding"), "speculative decoding")


if __name__ == "__main__":
    unittest.main()
