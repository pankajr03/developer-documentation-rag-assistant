import argparse
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from scripts.view_feedback import format_record, parse_limit
from services import feedback_service
from services.feedback_service import (
    MAX_COMMENT_LENGTH,
    MAX_EXCERPT_LENGTH,
    MAX_LIST_LIMIT,
    FeedbackNotFoundError,
    FeedbackStorageError,
    get_feedback_by_response_id,
    get_feedback_summary,
    initialize_feedback_database,
    list_recent_feedback,
    save_feedback,
    update_feedback,
)

QUESTION = "What is the reporting-time rule mentioned in the Speed Schedule?"
ANSWER = "Crew must report 45 minutes before departure [Source 1]."
SOURCES = [{
    "number": 1,
    "filename": "speed_schedule.pdf",
    "page": 3,
    "chunk_id": "speed_schedule_p3_c2",
    "document_id": "speed_schedule",
    "section": None,
    "content": "Section 3: Reporting Time. Crew must report 45 minutes early.",
    "distance": 0.18,
}]
CHUNKS = [{
    "text": "Section 3: Reporting Time. Crew must report 45 minutes early.",
    "source": "speed_schedule.pdf",
    "page": 3,
    "chunk_id": "speed_schedule_p3_c2",
    "chunk_index": 2,
    "distance": 0.18,
    "embedding": [0.1, 0.2],
}]


class FeedbackTestCase(unittest.TestCase):
    """Every test gets a throwaway database, never data/feedback.db."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "nested" / "feedback.db"

    def save(self, response_id="resp-1", rating=1, **overrides):
        arguments = dict(
            response_id=response_id, question=QUESTION, answer=ANSWER,
            rating=rating, sources=SOURCES, retrieved_chunks=CHUNKS,
            model_name="gpt-5.4-mini", retrieval_top_k=3,
            session_id="session-1", db_path=self.db,
        )
        arguments.update(overrides)
        return save_feedback(**arguments)

    def row_count(self):
        with closing(sqlite3.connect(self.db)) as connection:
            return connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]


class InitializeTests(FeedbackTestCase):
    def test_creates_directory_and_table(self):
        initialize_feedback_database(self.db)
        self.assertTrue(self.db.exists())
        with closing(sqlite3.connect(self.db)) as connection:
            columns = [row[1] for row in
                       connection.execute("PRAGMA table_info(feedback)")]
        self.assertEqual(columns, [
            "id", "response_id", "session_id", "question", "answer", "rating",
            "comment", "sources_json", "retrieved_chunks_json", "model_name",
            "retrieval_top_k", "created_at", "updated_at",
        ])

    def test_is_safe_to_call_twice_and_keeps_data(self):
        self.save()
        initialize_feedback_database(self.db)
        self.assertEqual(self.row_count(), 1)

    def test_unusable_database_path_raises_storage_error(self):
        # A directory cannot be opened as a database file.
        with self.assertRaises(FeedbackStorageError):
            initialize_feedback_database(Path(self.tmp.name))


class SaveFeedbackTests(FeedbackTestCase):
    def test_saves_thumbs_up(self):
        row = self.save(rating=1)
        self.assertEqual(row["rating"], 1)
        self.assertEqual(row["response_id"], "resp-1")
        self.assertEqual(row["question"], QUESTION)
        self.assertEqual(row["answer"], ANSWER)
        self.assertEqual(row["session_id"], "session-1")
        self.assertEqual(row["model_name"], "gpt-5.4-mini")
        self.assertEqual(row["retrieval_top_k"], 3)
        self.assertIsNotNone(row["created_at"])
        self.assertIsNone(row["updated_at"])

    def test_saves_thumbs_down(self):
        self.assertEqual(self.save(rating=-1)["rating"], -1)

    def test_stores_optional_comment(self):
        row = self.save(comment="  The answer was clear.  ")
        self.assertEqual(row["comment"], "The answer was clear.")

    def test_comment_is_optional(self):
        self.assertIsNone(self.save()["comment"])
        self.assertIsNone(self.save("resp-2", comment="   ")["comment"])

    def test_serializes_sources_as_json_with_truncated_excerpt(self):
        long_source = dict(SOURCES[0], content="x" * (MAX_EXCERPT_LENGTH * 3))
        row = self.save(sources=[long_source])
        sources = json.loads(row["sources_json"])
        self.assertEqual(sources[0]["filename"], "speed_schedule.pdf")
        self.assertEqual(sources[0]["page"], 3)
        self.assertEqual(sources[0]["chunk_id"], "speed_schedule_p3_c2")
        self.assertLessEqual(len(sources[0]["excerpt"]), MAX_EXCERPT_LENGTH + 1)
        self.assertNotIn("section", sources[0])  # missing metadata is omitted

    def test_retrieved_chunks_keep_metadata_but_not_text_or_embeddings(self):
        row = self.save()
        chunks = json.loads(row["retrieved_chunks_json"])
        self.assertEqual(chunks[0]["chunk_id"], "speed_schedule_p3_c2")
        self.assertNotIn("text", chunks[0])
        self.assertNotIn("embedding", chunks[0])
        self.assertNotIn("embedding", row["sources_json"])

    def test_no_sources_are_stored_as_empty_json_lists(self):
        row = self.save(sources=None, retrieved_chunks=None)
        self.assertEqual(json.loads(row["sources_json"]), [])
        self.assertEqual(json.loads(row["retrieved_chunks_json"]), [])

    def test_unicode_comment_round_trips(self):
        comment = "Très utile ✅ — 答案清楚，来源正确。 🚀"
        self.save(comment=comment)
        stored = get_feedback_by_response_id("resp-1", db_path=self.db)
        self.assertEqual(stored["comment"], comment)

    def test_control_characters_are_removed_from_comments(self):
        row = self.save(comment="ok\x00\x07 fine\nnext line")
        self.assertEqual(row["comment"], "ok fine\nnext line")

    def test_sql_in_comment_is_stored_as_text(self):
        comment = "'); DROP TABLE feedback; --"
        self.save(comment=comment)
        self.assertEqual(
            get_feedback_by_response_id("resp-1", db_path=self.db)["comment"],
            comment)
        self.assertEqual(self.row_count(), 1)


class UpsertTests(FeedbackTestCase):
    def test_changing_rating_updates_the_same_row(self):
        first = self.save(rating=1, comment="Good")
        second = self.save(rating=-1, comment="Actually wrong")
        self.assertEqual(self.row_count(), 1)
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["rating"], -1)
        self.assertEqual(second["comment"], "Actually wrong")

    def test_same_rating_twice_does_not_create_duplicates(self):
        self.save(rating=1)
        self.save(rating=1)
        self.save(rating=1)
        self.assertEqual(self.row_count(), 1)

    def test_different_responses_get_separate_rows(self):
        self.save("resp-1")
        self.save("resp-2")
        self.assertEqual(self.row_count(), 2)

    def test_created_at_is_preserved_and_updated_at_changes(self):
        times = iter(["2026-01-01T10:00:00+00:00", "2026-01-01T10:05:00+00:00"])
        with patch.object(feedback_service, "_utc_now", lambda: next(times)):
            first = self.save(rating=1)
            second = self.save(rating=-1)
        self.assertEqual(first["created_at"], "2026-01-01T10:00:00+00:00")
        self.assertIsNone(first["updated_at"])
        self.assertEqual(second["created_at"], first["created_at"])
        self.assertEqual(second["updated_at"], "2026-01-01T10:05:00+00:00")

    def test_update_keeps_original_answer_and_sources(self):
        first = self.save()
        second = self.save(answer="A different answer", sources=[])
        self.assertEqual(second["answer"], first["answer"])
        self.assertEqual(second["sources_json"], first["sources_json"])

    def test_database_rejects_duplicate_response_id_directly(self):
        self.save()
        with closing(sqlite3.connect(self.db)) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO feedback (response_id, question, answer, "
                    "rating, created_at) VALUES ('resp-1', 'q', 'a', 1, 't')")

    def test_database_rejects_invalid_rating_directly(self):
        initialize_feedback_database(self.db)
        with closing(sqlite3.connect(self.db)) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO feedback (response_id, question, answer, "
                    "rating, created_at) VALUES ('r', 'q', 'a', 5, 't')")


class UpdateFeedbackTests(FeedbackTestCase):
    def test_updates_rating_only(self):
        self.save(rating=1, comment="Keep me")
        row = update_feedback("resp-1", rating=-1, db_path=self.db)
        self.assertEqual(row["rating"], -1)
        self.assertEqual(row["comment"], "Keep me")
        self.assertIsNotNone(row["updated_at"])

    def test_updates_and_clears_comment(self):
        self.save(rating=1, comment="Old")
        row = update_feedback("resp-1", comment="New", db_path=self.db)
        self.assertEqual((row["rating"], row["comment"]), (1, "New"))
        row = update_feedback("resp-1", comment=None, db_path=self.db)
        self.assertIsNone(row["comment"])

    def test_missing_feedback_raises(self):
        with self.assertRaises(FeedbackNotFoundError):
            update_feedback("nope", rating=1, db_path=self.db)

    def test_requires_something_to_update(self):
        self.save()
        with self.assertRaises(ValueError):
            update_feedback("resp-1", db_path=self.db)

    def test_validates_values(self):
        self.save()
        with self.assertRaises(ValueError):
            update_feedback("resp-1", rating=0, db_path=self.db)
        with self.assertRaises(ValueError):
            update_feedback("resp-1", comment="x" * (MAX_COMMENT_LENGTH + 1),
                            db_path=self.db)


class ValidationTests(FeedbackTestCase):
    def test_rejects_invalid_ratings(self):
        for rating in (0, 2, -2, "1", None, 1.5, True):
            with self.subTest(rating=rating), self.assertRaises(ValueError):
                self.save(rating=rating)
        self.assertFalse(self.db.exists())

    def test_rejects_missing_response_id(self):
        for response_id in (None, "", "   ", 123):
            with self.subTest(response_id=response_id), \
                    self.assertRaises(ValueError):
                self.save(response_id=response_id)

    def test_rejects_missing_question_or_answer(self):
        with self.assertRaises(ValueError):
            self.save(question="  ")
        with self.assertRaises(ValueError):
            self.save(answer="")

    def test_comment_at_the_limit_is_accepted(self):
        row = self.save(comment="a" * MAX_COMMENT_LENGTH)
        self.assertEqual(len(row["comment"]), MAX_COMMENT_LENGTH)

    def test_rejects_comment_over_the_limit(self):
        with self.assertRaises(ValueError):
            self.save(comment="a" * (MAX_COMMENT_LENGTH + 1))
        self.assertFalse(self.db.exists())

    def test_rejects_invalid_top_k(self):
        for top_k in (0, -1, "3", True):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.save(retrieval_top_k=top_k)

    def test_long_question_and_answer_are_truncated(self):
        row = self.save(question="q" * 10_000, answer="a" * 50_000)
        self.assertLessEqual(len(row["question"]), feedback_service.MAX_QUESTION_LENGTH + 1)
        self.assertLessEqual(len(row["answer"]), feedback_service.MAX_ANSWER_LENGTH + 1)


class ReadTests(FeedbackTestCase):
    def test_unknown_response_returns_none(self):
        self.assertIsNone(get_feedback_by_response_id("nope", db_path=self.db))

    def test_feedback_survives_a_new_connection(self):
        self.save(rating=-1, comment="Not in the docs.")
        # A fresh call opens a new connection, like an app restart would.
        stored = get_feedback_by_response_id("resp-1", db_path=self.db)
        self.assertEqual((stored["rating"], stored["comment"]),
                         (-1, "Not in the docs."))

    def test_get_requires_response_id(self):
        with self.assertRaises(ValueError):
            get_feedback_by_response_id("", db_path=self.db)


class SummaryTests(FeedbackTestCase):
    def test_empty_database(self):
        self.assertEqual(get_feedback_summary(self.db), {
            "total": 0, "thumbs_up": 0, "thumbs_down": 0,
            "positive_percentage": 0.0,
        })

    def test_counts_and_percentage(self):
        self.save("a", rating=1)
        self.save("b", rating=1)
        self.save("c", rating=1)
        self.save("d", rating=-1)
        self.assertEqual(get_feedback_summary(self.db), {
            "total": 4, "thumbs_up": 3, "thumbs_down": 1,
            "positive_percentage": 75.0,
        })

    def test_changed_rating_is_counted_once(self):
        self.save("a", rating=1)
        self.save("a", rating=-1)
        summary = get_feedback_summary(self.db)
        self.assertEqual((summary["total"], summary["thumbs_up"],
                          summary["thumbs_down"]), (1, 0, 1))
        self.assertEqual(summary["positive_percentage"], 0.0)


class ListRecentTests(FeedbackTestCase):
    def test_returns_newest_first_with_safe_fields_only(self):
        self.save("a", comment="first")
        self.save("b", rating=-1, comment="second")
        records = list_recent_feedback(10, db_path=self.db)
        self.assertEqual([r["response_id"] for r in records], ["b", "a"])
        self.assertEqual(records[0]["source_filenames"], ["speed_schedule.pdf"])
        for private in ("answer", "sources_json", "retrieved_chunks_json"):
            self.assertNotIn(private, records[0])

    def test_respects_limit(self):
        for index in range(5):
            self.save(f"r{index}")
        self.assertEqual(len(list_recent_feedback(2, db_path=self.db)), 2)

    def test_rejects_invalid_limits(self):
        for limit in (0, -1, MAX_LIST_LIMIT + 1, "5", True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                list_recent_feedback(limit, db_path=self.db)


class DatabaseErrorTests(FeedbackTestCase):
    def test_sqlite_errors_become_storage_errors(self):
        with patch("services.feedback_service.sqlite3.connect",
                   side_effect=sqlite3.OperationalError("disk I/O error")):
            with self.assertRaises(FeedbackStorageError):
                self.save()
            with self.assertRaises(FeedbackStorageError):
                get_feedback_by_response_id("resp-1", db_path=self.db)
            with self.assertRaises(FeedbackStorageError):
                get_feedback_summary(self.db)
            with self.assertRaises(FeedbackStorageError):
                update_feedback("resp-1", rating=1, db_path=self.db)
            with self.assertRaises(FeedbackStorageError):
                list_recent_feedback(5, db_path=self.db)

    def test_storage_error_message_hides_sql_details(self):
        with patch("services.feedback_service.sqlite3.connect",
                   side_effect=sqlite3.OperationalError("secret path detail")):
            with self.assertRaises(FeedbackStorageError) as context:
                self.save()
        self.assertNotIn("secret path detail", str(context.exception))


class ViewFeedbackScriptTests(FeedbackTestCase):
    def test_parse_limit(self):
        self.assertEqual(parse_limit("20"), 20)
        for bad in ("0", "-3", str(MAX_LIST_LIMIT + 1), "abc", "2.5"):
            with self.subTest(value=bad), \
                    self.assertRaises(argparse.ArgumentTypeError):
                parse_limit(bad)

    def test_format_record_shows_useful_fields_only(self):
        self.save(comment="Clear answer.")
        record = list_recent_feedback(1, db_path=self.db)[0]
        text = format_record(record)
        for expected in ("ID:", "Response ID:", "resp-1", QUESTION,
                         "thumbs up", "Clear answer.", "Created at:",
                         "Updated at:", "speed_schedule.pdf"):
            self.assertIn(expected, text)
        self.assertNotIn(ANSWER, text)
        self.assertNotIn("Crew must report", text)


if __name__ == "__main__":
    unittest.main()
