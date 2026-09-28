"""Tests for ui.services with fakes. No OpenAI calls, no production data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import openai

from rag.generator import FALLBACK_MESSAGE, build_citations, build_sources
from services.feedback_service import (
    FeedbackStorageError,
    get_feedback_by_response_id,
)
from ui import services
from ui.services import AssistantError

CHUNKS = [
    {"text": "Report 30 minutes before race time.", "source": "Schedule.pdf",
     "document_id": "schedule", "chunk_id": "schedule_p1_c0", "chunk_index": 0,
     "page": 1, "section": None, "distance": 0.31},
    {"text": "One lap is 200m.", "source": "Schedule.pdf",
     "document_id": "schedule", "chunk_id": "schedule_p2_c1", "chunk_index": 1,
     "page": 2, "section": None, "distance": 0.52},
]


def fake_generate(answer, unverified=()):
    def generate(question, chunks, top_k=None):
        sources = build_sources(chunks)
        return {"response_id": "resp-1", "question": question, "answer": answer,
                "sources": sources, "citations": build_citations(answer, sources),
                "unverified_citation_numbers": list(unverified),
                "retrieved_chunks": chunks, "model_name": "fake-model",
                "retrieval_top_k": top_k}
    return generate


def openai_error(cls):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    if cls is openai.APITimeoutError:
        return cls(request=request)
    if cls is openai.APIConnectionError:
        return cls(request=request)
    response = httpx.Response(401 if cls is openai.AuthenticationError else 429,
                              request=request)
    return cls("Incorrect API key provided: sk-abc***xyz", response=response, body=None)


class FakeCollection:
    def __init__(self, metadatas):
        self.metadatas = metadatas

    def count(self):
        return len(self.metadatas)

    def get(self, include=None):
        return {"ids": [str(i) for i in range(len(self.metadatas))],
                "metadatas": self.metadatas}


class FakeClient:
    def __init__(self, collection=None, error=None):
        self.collection, self.error = collection, error

    def get_collection(self, name):
        if self.error:
            raise self.error
        if self.collection is None:
            raise ValueError(f"Collection [{name}] does not exist")
        return self.collection


class KnowledgeBaseStatusTests(unittest.TestCase):
    def test_missing_folder_is_empty(self):
        status = services.knowledge_base_status(None)
        self.assertEqual(status.state, "empty")
        self.assertEqual(status.chunk_count, 0)

    def test_missing_collection_is_empty(self):
        self.assertEqual(services.knowledge_base_status(FakeClient()).state, "empty")

    def test_empty_collection_is_empty(self):
        status = services.knowledge_base_status(FakeClient(FakeCollection([])))
        self.assertEqual(status.state, "empty")

    def test_ready_collection_counts_documents(self):
        metadatas = [
            {"source": "Schedule.pdf", "document_id": "schedule", "content_hash": "h1"},
            {"source": "Schedule.pdf", "document_id": "schedule", "content_hash": "h1"},
            {"source": r"C:\private\Guide.md", "document_id": "guide"},
        ]
        status = services.knowledge_base_status(FakeClient(FakeCollection(metadatas)))
        self.assertEqual(status.state, "ready")
        self.assertEqual(status.chunk_count, 3)
        self.assertEqual(status.document_count, 2)
        self.assertEqual([(d.name, d.chunks) for d in status.documents],
                         [("Guide.md", 1), ("Schedule.pdf", 2)])
        self.assertEqual(status.document_ids, {"schedule", "guide"})
        self.assertEqual(status.content_hashes, {"h1"})

    def test_large_collection_does_not_claim_a_document_count(self):
        collection = FakeCollection([{}] * 3)
        with patch("ui.services.MAX_STATUS_SCAN", 2):
            status = services.knowledge_base_status(FakeClient(collection))
        self.assertEqual(status.state, "ready")
        self.assertIsNone(status.document_count)

    def test_database_failure_is_an_error_state(self):
        status = services.knowledge_base_status(
            FakeClient(error=RuntimeError("disk I/O error at C:\\secret")))
        self.assertEqual(status.state, "error")
        self.assertNotIn("secret", status.message)


class AskQuestionTests(unittest.TestCase):
    def test_answer_with_citations(self):
        retrieve = MagicMock(return_value=CHUNKS)
        answer = services.ask_question(
            "When do I report?", 3, retrieve=retrieve,
            generate=fake_generate("Report 30 minutes early [Source 1]."))
        retrieve.assert_called_once_with("When do I report?", top_k=3)
        self.assertEqual(answer["response_id"], "resp-1")
        self.assertFalse(answer["is_fallback"])
        self.assertEqual(answer["warnings"], [])
        self.assertEqual([s["cited"] for s in answer["sources"]], [True, False])
        # Feedback data keeps metadata but never the full chunk text.
        chunk = answer["feedback_payload"]["retrieved_chunks"][0]
        self.assertNotIn("text", chunk)
        self.assertEqual(chunk["chunk_id"], "schedule_p1_c0")

    def test_fallback_answer(self):
        answer = services.ask_question(
            "Where is the venue?", 3, retrieve=lambda q, top_k: CHUNKS,
            generate=fake_generate(FALLBACK_MESSAGE))
        self.assertTrue(answer["is_fallback"])
        self.assertEqual(answer["warnings"], [])
        self.assertFalse(any(s["cited"] for s in answer["sources"]))

    def test_warnings_for_uncited_and_removed_citations(self):
        answer = services.ask_question(
            "q", 3, retrieve=lambda q, top_k: CHUNKS,
            generate=fake_generate("Some claim.", unverified=[7]))
        self.assertEqual(len(answer["warnings"]), 2)

    def test_empty_knowledge_base(self):
        def retrieve(question, top_k):
            raise ValueError("No documentation has been indexed yet. Upload ...")

        with self.assertRaises(AssistantError) as caught:
            services.ask_question("q", 3, retrieve=retrieve, generate=MagicMock())
        self.assertEqual(str(caught.exception), services.MSG_EMPTY_KB)

    def test_generation_failure_is_user_safe(self):
        def generate(*args, **kwargs):
            raise RuntimeError("Traceback ... C:\\Users\\me\\.env sk-live-123")

        with self.assertRaises(AssistantError) as caught:
            services.ask_question("q", 3, retrieve=lambda q, top_k: CHUNKS,
                                  generate=generate)
        message = str(caught.exception)
        self.assertIn("Could not generate an answer", message)
        self.assertNotIn("sk-", message)
        self.assertNotIn("Users", message)


class FriendlyErrorTests(unittest.TestCase):
    def wrapped(self, cause):
        try:
            try:
                raise cause
            except Exception as error:
                raise RuntimeError("The OpenAI request failed.") from error
        except RuntimeError as outer:
            return outer

    def test_openai_errors_are_mapped_through_the_cause_chain(self):
        cases = {
            openai.AuthenticationError: services.MSG_INVALID_KEY,
            openai.RateLimitError: services.MSG_RATE_LIMIT,
            openai.APITimeoutError: services.MSG_TIMEOUT,
            openai.APIConnectionError: services.MSG_NETWORK,
        }
        for cls, expected in cases.items():
            with self.subTest(cls=cls.__name__):
                message = services.friendly_error(
                    self.wrapped(openai_error(cls)), "fallback")
                self.assertEqual(message, expected)
                self.assertNotIn("sk-", message)

    def test_missing_api_key(self):
        error = ValueError("OPENAI_API_KEY is not set. Add it to the project's .env file.")
        self.assertEqual(services.friendly_error(error, "x"), services.MSG_MISSING_KEY)

    def test_feedback_storage_error(self):
        self.assertEqual(
            services.friendly_error(FeedbackStorageError("sqlite locked"), "x"),
            services.MSG_FEEDBACK_DB)

    def test_unknown_error_uses_fallback(self):
        self.assertEqual(
            services.friendly_error(KeyError("/home/me/private"), "Try again."),
            "Try again.")

    def test_logs_only_exception_types(self):
        with self.assertLogs("ui.services", level="WARNING") as logs:
            services._log_failure("Retrieval", RuntimeError("sk-secret-value"))
        self.assertNotIn("sk-secret", "".join(logs.output))
        self.assertIn("RuntimeError", "".join(logs.output))


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "feedback.db"
        answer = services.ask_question(
            "When do I report?", 3, retrieve=lambda q, top_k: CHUNKS,
            generate=fake_generate("Report early [Source 1]."))
        self.message = {"role": "assistant", "kind": "answer", **answer}

    def save(self):
        """save_feedback bound to a temporary database."""
        from services.feedback_service import save_feedback

        return lambda **fields: save_feedback(db_path=self.db, **fields)

    def test_saves_rating_for_the_stored_answer(self):
        services.save_answer_feedback(self.message, 1, "  Clear  ", "session-1",
                                      save=self.save())
        row = get_feedback_by_response_id("resp-1", db_path=self.db)
        self.assertEqual(row["rating"], 1)
        self.assertEqual(row["comment"], "Clear")
        self.assertEqual(row["question"], "When do I report?")
        self.assertEqual(row["model_name"], "fake-model")
        self.assertIn("schedule_p1_c0", row["retrieved_chunks_json"])

    def test_invalid_comment_message_is_shown(self):
        with self.assertRaisesRegex(AssistantError, "at most"):
            services.save_answer_feedback(self.message, 1, "x" * 5000, "s",
                                          save=self.save())

    def test_database_failure_is_user_safe(self):
        def failing(**fields):
            raise FeedbackStorageError("Could not save your feedback.")

        with self.assertRaises(AssistantError) as caught:
            services.save_answer_feedback(self.message, -1, "", "s", save=failing)
        self.assertEqual(str(caught.exception), services.MSG_FEEDBACK_DB)

    def test_empty_feedback_overview(self):
        from services import feedback_service

        summary, rows = services.feedback_overview(
            10,
            summary_fn=lambda: feedback_service.get_feedback_summary(db_path=self.db),
            list_fn=lambda limit: feedback_service.list_recent_feedback(
                limit, db_path=self.db))
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["positive_percentage"], 0.0)
        self.assertEqual(rows, [])

    def test_feedback_overview_rows(self):
        records = [{"response_id": "r1", "question": "q " * 100, "rating": -1,
                    "comment": None, "source_filenames": ["a.pdf"],
                    "created_at": "2026-09-28T10:11:12+00:00", "updated_at": None}]
        _, rows = services.feedback_overview(
            5, summary_fn=lambda: {"total": 1}, list_fn=lambda limit: records)
        self.assertEqual(rows[0]["Rating"], "👎 Not helpful")
        self.assertEqual(rows[0]["Date (UTC)"], "2026-09-28 10:11")
        self.assertEqual(rows[0]["Comment"], "")
        self.assertTrue(rows[0]["Question"].endswith("…"))


class IndexDocumentsTests(unittest.TestCase):
    def setUp(self):
        self.embed = MagicMock(side_effect=lambda texts: [[0.1]] * len(texts))
        self.store = MagicMock(side_effect=lambda chunks, embeddings, source, **kw:
                               len(chunks))

    def index(self, files, **kwargs):
        return services.index_documents(files, embed=self.embed, store=self.store,
                                        **kwargs)

    def test_indexes_a_text_file(self):
        stages = []
        report = self.index([("guide.md", b"# Setup\nInstall it.")],
                            progress=lambda stage, *a: stages.append(stage))
        outcome = report.outcomes[0]
        self.assertEqual((outcome.status, outcome.chunks), ("indexed", 1))
        _, kwargs = self.store.call_args
        self.assertEqual(kwargs["document_id"], "guide")
        self.assertEqual(len(kwargs["content_hash"]), 64)
        self.assertEqual(stages, ["Reading document", "Creating chunks",
                                  "Generating embeddings", "Saving to ChromaDB",
                                  "Completed"])

    def test_invalid_files_are_skipped_before_any_api_call(self):
        report = self.index([("run.py", b"print(1)"), ("empty.txt", b"")])
        self.assertEqual([o.status for o in report.outcomes], ["skipped", "skipped"])
        self.embed.assert_not_called()

    def test_duplicates_are_skipped_not_replaced(self):
        report = self.index(
            [("a.txt", b"same text"), ("copy of a.txt", b"same text"),
             ("guide.md", b"other"), ("GUIDE.txt", b"different")],
            known_document_ids={"existing"})
        self.assertEqual([o.status for o in report.outcomes],
                         ["indexed", "skipped", "indexed", "skipped"])
        self.assertIn("same content", report.outcomes[1].message)
        self.assertIn("'guide'", report.outcomes[3].message)
        self.assertEqual(self.store.call_count, 2)

    def test_already_indexed_document_is_skipped(self):
        report = self.index([("Schedule.pdf", b"%PDF-1.4")],
                            known_document_ids={"schedule"})
        self.assertEqual(report.outcomes[0].status, "skipped")
        self.embed.assert_not_called()

    def test_extraction_failure(self):
        report = self.index([("broken.pdf", b"not really a pdf"),
                             ("latin1.txt", "café".encode("latin-1"))])
        self.assertEqual([o.status for o in report.outcomes], ["failed", "failed"])
        self.assertIn("Could not extract text", report.outcomes[0].message)
        self.assertIn("UTF-8", report.outcomes[1].message)

    def test_no_usable_text(self):
        report = self.index([("blank.txt", b"   \n\n  ")])
        self.assertEqual(report.outcomes[0].status, "skipped")
        self.assertIn("No usable text", report.outcomes[0].message)

    def test_embedding_failure_does_not_stop_other_files(self):
        self.embed.side_effect = [
            ValueError("OPENAI_API_KEY is not set."), [[0.1]]]
        report = self.index([("a.txt", b"first"), ("b.txt", b"second")])
        self.assertEqual([o.status for o in report.outcomes], ["failed", "indexed"])
        self.assertEqual(report.outcomes[0].message, services.MSG_MISSING_KEY)
        self.assertEqual(report.chunks_created, 1)
        self.assertEqual(report.count("failed"), 1)

    def test_embeddings_are_requested_in_batches(self):
        with patch("ui.services.EMBEDDING_BATCH_SIZE", 2):
            self.index([("big.txt", ("word " * 1000).encode())])
        self.assertGreater(self.embed.call_count, 1)
        self.assertTrue(all(len(c.args[0]) <= 2 for c in self.embed.call_args_list))


if __name__ == "__main__":
    unittest.main()
