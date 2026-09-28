import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag.chunker import chunk_sections, chunk_text
from rag.loader import document_id_from_name, load_document, section_text
from rag.vector_store import build_chunk_id, store_chunks


def _uploaded(name: str, data: bytes) -> SimpleNamespace:
    return SimpleNamespace(name=name, getvalue=lambda: data)


class LoaderTests(unittest.TestCase):
    def test_text_file_has_no_page_number(self):
        sections = load_document(_uploaded("guide.md", b"# Title\nBody"))
        self.assertEqual(sections, [{"text": "# Title\nBody", "page": None}])

    @patch("rag.loader.PdfReader")
    def test_pdf_sections_are_numbered_from_one(self, reader):
        reader.return_value = SimpleNamespace(pages=[
            SimpleNamespace(extract_text=lambda: "page one"),
            SimpleNamespace(extract_text=lambda: "   "),
            SimpleNamespace(extract_text=lambda: "page three"),
        ])
        sections = load_document(_uploaded("speed_schedule.pdf", b"%PDF-"))
        # The empty page is dropped, but page numbers stay true to the file.
        self.assertEqual(sections, [
            {"text": "page one", "page": 1},
            {"text": "page three", "page": 3},
        ])

    def test_unsupported_extension_raises(self):
        with self.assertRaises(ValueError):
            load_document(_uploaded("notes.docx", b""))

    def test_section_text_joins_pages(self):
        self.assertEqual(
            section_text([{"text": "a", "page": 1}, {"text": "b", "page": 2}]),
            "a\nb",
        )

    def test_document_id_is_slugified(self):
        self.assertEqual(
            document_id_from_name("Speed Schedule 2026.pdf"), "speed_schedule_2026")


class ChunkSectionsTests(unittest.TestCase):
    def test_keeps_page_and_numbers_chunks_across_sections(self):
        sections = [
            {"text": "a" * 10, "page": 1},
            {"text": "b" * 10, "page": 2},
        ]
        chunks = chunk_sections(sections, chunk_size=5, overlap=0)
        self.assertEqual([c["chunk_index"] for c in chunks], [0, 1, 2, 3])
        self.assertEqual([c["page"] for c in chunks], [1, 1, 2, 2])

    def test_captures_markdown_heading_as_section(self):
        text = "# Intro\nhello\n\n## Reporting Time\nreport 45 minutes early"
        chunks = chunk_sections([{"text": text, "page": None}],
                                chunk_size=20, overlap=0)
        self.assertEqual(chunks[0]["section"], "Intro")
        self.assertEqual(chunks[-1]["section"], "Reporting Time")

    def test_section_is_none_without_headings(self):
        chunks = chunk_sections([{"text": "plain text", "page": 4}])
        self.assertIsNone(chunks[0]["section"])
        self.assertEqual(chunks[0]["page"], 4)

    def test_chunk_text_still_returns_plain_strings(self):
        self.assertEqual(chunk_text("abcdef", chunk_size=3, overlap=0),
                         ["abc", "def"])


class StoreChunksTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("rag.vector_store._get_collection")
        self.get_collection = patcher.start()
        self.addCleanup(patcher.stop)
        self.collection = MagicMock()
        self.get_collection.return_value = self.collection

    def _upsert_kwargs(self):
        return self.collection.upsert.call_args.kwargs

    def test_stores_full_metadata_and_unique_ids(self):
        chunks = [
            {"text": "first", "page": 3, "section": "Reporting Time",
             "chunk_index": 2},
            {"text": "second", "page": 3, "section": "Reporting Time",
             "chunk_index": 3},
        ]
        stored = store_chunks(chunks, [[0.1], [0.2]], "speed_schedule.pdf")

        self.assertEqual(stored, 2)
        kwargs = self._upsert_kwargs()
        self.assertEqual(kwargs["ids"],
                         ["speed_schedule_p3_c2", "speed_schedule_p3_c3"])
        self.assertEqual(kwargs["metadatas"][0], {
            "source": "speed_schedule.pdf",
            "document_id": "speed_schedule",
            "chunk_id": "speed_schedule_p3_c2",
            "chunk_index": 2,
            "page": 3,
            "section": "Reporting Time",
        })

    def test_omits_missing_optional_metadata(self):
        store_chunks([{"text": "body", "page": None, "section": None,
                       "chunk_index": 0}], [[0.1]], "guide.md")
        metadata = self._upsert_kwargs()["metadatas"][0]
        self.assertNotIn("page", metadata)
        self.assertNotIn("section", metadata)
        self.assertEqual(metadata["chunk_id"], "guide_c0")

    def test_stores_content_hash_only_when_given(self):
        store_chunks(["one"], [[0.1]], "guide.md")
        self.assertNotIn("content_hash",
                         self.collection.upsert.call_args.kwargs["metadatas"][0])
        store_chunks(["one"], [[0.1]], "guide.md", content_hash="abc123")
        self.assertEqual(
            self.collection.upsert.call_args.kwargs["metadatas"][0]["content_hash"],
            "abc123")

    def test_accepts_plain_string_chunks(self):
        store_chunks(["one", "two"], [[0.1], [0.2]], "guide.md")
        self.assertEqual(self._upsert_kwargs()["ids"], ["guide_c0", "guide_c1"])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            store_chunks([{"text": "a"}], [[0.1], [0.2]], "guide.md")

    def test_build_chunk_id_without_page(self):
        self.assertEqual(build_chunk_id("guide", None, 5), "guide_c5")


if __name__ == "__main__":
    unittest.main()
