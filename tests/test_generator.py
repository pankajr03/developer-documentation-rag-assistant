import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.generator import NOT_FOUND_MESSAGE, build_context, generate_answer

CHUNKS = [
    {"text": "Access tokens expire after 60 minutes.",
     "source": "guide.md", "chunk_index": 7, "distance": 0.21},
    {"text": "Refresh tokens remain valid for 30 days.",
     "source": "guide.md", "chunk_index": 8, "distance": 0.34},
]


class BuildContextTests(unittest.TestCase):
    def test_includes_source_metadata_and_text(self):
        context = build_context(CHUNKS)
        self.assertIn("[SOURCE 1]\nFile: guide.md\nChunk: 7", context)
        self.assertIn("[SOURCE 2]\nFile: guide.md\nChunk: 8", context)
        self.assertIn("Refresh tokens remain valid for 30 days.", context)

    def test_excludes_distances(self):
        self.assertNotIn("0.21", build_context(CHUNKS))


class GenerateAnswerTests(unittest.TestCase):
    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            generate_answer("   ", CHUNKS)

    @patch("rag.generator._create_client")
    def test_no_chunks_returns_fallback_without_calling_llm(self, create_client):
        self.assertEqual(generate_answer("Uses Redis?", []), NOT_FOUND_MESSAGE)
        create_client.assert_not_called()

    @patch("rag.generator._create_client")
    def test_returns_model_answer(self, create_client):
        create_client.return_value.responses.create.return_value = SimpleNamespace(
            status="completed", output_text=" Tokens expire after 60 minutes. ")
        answer = generate_answer("When do tokens expire?", CHUNKS)
        self.assertEqual(answer, "Tokens expire after 60 minutes.")

    @patch("rag.generator._create_client")
    def test_empty_model_response_raises(self, create_client):
        create_client.return_value.responses.create.return_value = SimpleNamespace(
            status="completed", output_text="")
        with self.assertRaises(RuntimeError):
            generate_answer("When do tokens expire?", CHUNKS)


if __name__ == "__main__":
    unittest.main()
