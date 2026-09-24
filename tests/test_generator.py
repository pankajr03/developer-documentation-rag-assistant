import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.generator import (
    FALLBACK_MESSAGE,
    build_context,
    build_sources,
    generate_answer,
)

CHUNKS = [
    {
        "text": "Crew must report 45 minutes before departure.",
        "source": "speed_schedule.pdf",
        "document_id": "speed_schedule",
        "chunk_id": "speed_schedule_p3_c2",
        "chunk_index": 2,
        "page": 3,
        "section": "Reporting Time",
        "distance": 0.18,
    },
    {
        "text": "Late reports are recorded by the duty officer.",
        "source": "speed_schedule.pdf",
        "document_id": "speed_schedule",
        "chunk_id": "speed_schedule_p3_c3",
        "chunk_index": 3,
        "page": 3,
        "section": "Reporting Time",
        "distance": 0.31,
    },
]


def _response(text: str, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(status=status, output_text=text)


class BuildSourcesTests(unittest.TestCase):
    def test_preserves_source_metadata(self):
        source = build_sources(CHUNKS)[0]
        self.assertEqual(source["filename"], "speed_schedule.pdf")
        self.assertEqual(source["page"], 3)
        self.assertEqual(source["chunk_id"], "speed_schedule_p3_c2")
        self.assertEqual(source["document_id"], "speed_schedule")
        self.assertEqual(source["section"], "Reporting Time")
        self.assertEqual(source["distance"], 0.18)
        self.assertEqual(source["content"], CHUNKS[0]["text"])

    def test_numbering_is_stable_and_sequential(self):
        numbers = [source["number"] for source in build_sources(CHUNKS)]
        self.assertEqual(numbers, [1, 2])
        self.assertEqual(build_sources(CHUNKS), build_sources(CHUNKS))

    def test_removes_duplicate_chunks(self):
        sources = build_sources([CHUNKS[0], dict(CHUNKS[0]), CHUNKS[1]])
        self.assertEqual([s["chunk_id"] for s in sources],
                         ["speed_schedule_p3_c2", "speed_schedule_p3_c3"])

    def test_keeps_different_chunks_from_the_same_page(self):
        sources = build_sources(CHUNKS)
        self.assertEqual(len(sources), 2)
        self.assertEqual({s["page"] for s in sources}, {3})

    def test_missing_page_metadata_stays_none(self):
        chunk = {"text": "Setup notes.", "source": "guide.md",
                 "chunk_id": "guide_c0", "distance": 0.4}
        source = build_sources([chunk])[0]
        self.assertIsNone(source["page"])
        self.assertIsNone(source["section"])


class BuildContextTests(unittest.TestCase):
    def test_numbers_sources_and_includes_metadata(self):
        context = build_context(build_sources(CHUNKS))
        self.assertIn("[Source 1]", context)
        self.assertIn("File: speed_schedule.pdf", context)
        self.assertIn("Page: 3", context)
        self.assertIn("Chunk ID: speed_schedule_p3_c2", context)
        self.assertIn("Crew must report 45 minutes before departure.", context)

    def test_omits_missing_page_instead_of_inventing_one(self):
        sources = build_sources([{"text": "Setup notes.", "source": "guide.md",
                                  "chunk_id": "guide_c0", "distance": 0.4}])
        self.assertNotIn("Page:", build_context(sources))

    def test_excludes_distance(self):
        self.assertNotIn("0.18", build_context(build_sources(CHUNKS)))


class GenerateAnswerTests(unittest.TestCase):
    def test_empty_question_raises(self):
        with self.assertRaises(ValueError):
            generate_answer("   ", CHUNKS)

    @patch("rag.generator._create_client")
    def test_no_chunks_returns_fallback_without_calling_llm(self, create_client):
        result = generate_answer("Does it use Redis?", [])
        self.assertEqual(result["answer"], FALLBACK_MESSAGE)
        self.assertEqual(result["sources"], [])
        create_client.assert_not_called()

    @patch("rag.generator._create_client")
    def test_returns_structured_result_with_citation(self, create_client):
        create_client.return_value.responses.create.return_value = _response(
            "Crew report 45 minutes before departure [Source 1].")
        result = generate_answer("What is the reporting-time rule?", CHUNKS)

        self.assertEqual(set(result), {"question", "answer", "sources"})
        self.assertEqual(result["question"], "What is the reporting-time rule?")
        self.assertIn("[Source 1]", result["answer"])
        self.assertEqual(len(result["sources"]), 2)
        self.assertNotIn("embedding", str(result["sources"]))

    @patch("rag.generator._create_client")
    def test_unsupported_question_returns_fallback_message(self, create_client):
        create_client.return_value.responses.create.return_value = _response(
            FALLBACK_MESSAGE)
        result = generate_answer("Does it use Redis?", CHUNKS)
        self.assertEqual(result["answer"], FALLBACK_MESSAGE)
        # Searched sources are still reported back to the user.
        self.assertEqual(len(result["sources"]), 2)

    @patch("rag.generator._create_client")
    def test_strips_citations_for_sources_that_do_not_exist(self, create_client):
        create_client.return_value.responses.create.return_value = _response(
            "Crew report 45 minutes early [Source 1] [Source 7].")
        answer = generate_answer("Reporting time?", CHUNKS)["answer"]
        self.assertIn("[Source 1]", answer)
        self.assertNotIn("[Source 7]", answer)

    @patch("rag.generator._create_client")
    def test_empty_model_response_raises(self, create_client):
        create_client.return_value.responses.create.return_value = _response("")
        with self.assertRaises(RuntimeError):
            generate_answer("Reporting time?", CHUNKS)

    @patch("rag.generator._create_client")
    def test_incomplete_model_response_raises(self, create_client):
        create_client.return_value.responses.create.return_value = _response(
            "partial", status="incomplete")
        with self.assertRaises(RuntimeError):
            generate_answer("Reporting time?", CHUNKS)


if __name__ == "__main__":
    unittest.main()
