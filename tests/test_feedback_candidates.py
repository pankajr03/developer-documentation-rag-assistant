import contextlib
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from evaluation.candidates import (
    candidate_id,
    normalize_question,
)
from scripts import export_feedback_candidates as exporter
from services.feedback_service import save_feedback

SOURCES = [{"number": 1, "filename": "speed_schedule.pdf", "chunk_id": "speed_p1_c0",
            "content": "PRIVATE EXCERPT TEXT"}]


class ExporterTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "feedback.db"
        self.output = self.root / "out" / "candidates.jsonl"
        self.golden = self.root / "golden.jsonl"  # deliberately absent by default

    def rate(self, response_id, question, rating, comment=None):
        save_feedback(response_id=response_id, question=question,
                      answer="A SECRET ANSWER", rating=rating, comment=comment,
                      sources=SOURCES, retrieved_chunks=[], model_name="m",
                      retrieval_top_k=3, session_id="SESSION-SECRET", db_path=self.db)

    def run_export(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = exporter.main(["--db", str(self.db), "--output", str(self.output),
                                  "--golden", str(self.golden), *extra])
        return code, out.getvalue(), err.getvalue()

    def read_output(self):
        if not self.output.exists():
            return []
        return [json.loads(line) for line in
                self.output.read_text(encoding="utf-8").splitlines() if line.strip()]


class NormalizationTests(unittest.TestCase):
    def test_normalize_question(self):
        self.assertEqual(normalize_question("  What IS   the fee?? "), "what is the fee")

    def test_candidate_id_is_stable_across_wording_noise(self):
        self.assertEqual(candidate_id("What is the fee?"), candidate_id("what is  the FEE"))
        self.assertNotEqual(candidate_id("What is the fee?"), candidate_id("Who won?"))
        self.assertRegex(candidate_id("x"), r"^cand_[0-9a-f]{12}$")


class ExportTests(ExporterTestCase):
    def test_exports_only_thumbs_down_and_marks_unreviewed(self):
        self.rate("r1", "Where is the venue?", -1, "Information was unavailable.")
        self.rate("r2", "What is the reporting time?", 1)
        code, out, _ = self.run_export()
        self.assertEqual(code, 0)
        rows = self.read_output()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["question"], "Where is the venue?")
        self.assertEqual(row["review_status"], "unreviewed")
        self.assertEqual(row["feedback_comments"], ["Information was unavailable."])
        self.assertEqual(row["retrieved_source_ids"], ["speed_p1_c0"])
        self.assertIn("unreviewed", out)

    def test_excludes_sensitive_data(self):
        self.rate("r1", "Where is the venue?", -1, "bad")
        self.run_export()
        text = self.output.read_text(encoding="utf-8")
        for secret in ("SESSION-SECRET", "A SECRET ANSWER", "PRIVATE EXCERPT TEXT", "session_id"):
            self.assertNotIn(secret, text)

    def test_does_not_modify_the_feedback_database(self):
        self.rate("r1", "Where is the venue?", -1)
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        self.run_export()
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 1)

    def test_same_question_is_merged_into_one_candidate(self):
        self.rate("r1", "Where is the venue?", -1, "first")
        self.rate("r2", "where is the  VENUE", -1, "second")
        self.run_export()
        rows = self.read_output()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["feedback_count"], 2)
        self.assertEqual(rows[0]["feedback_comments"], ["first", "second"])

    def test_rerun_does_not_duplicate_or_overwrite_reviewed_rows(self):
        self.rate("r1", "Where is the venue?", -1)
        self.run_export()
        # A reviewer edits the file, then a new complaint arrives.
        rows = self.read_output()
        rows[0]["review_notes"] = "reviewed by hand"
        self.output.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        self.rate("r2", "What is the entry fee?", -1)
        code, out, _ = self.run_export()
        self.assertEqual(code, 0)
        rows = self.read_output()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["review_notes"], "reviewed by hand")
        self.run_export()
        self.assertEqual(len(self.read_output()), 2)

    def test_questions_already_in_golden_dataset_are_skipped(self):
        golden_case = {"id": "g1", "question": "Where is the venue?", "expected_answer": "n/a",
                       "expected_keywords": [], "expected_source_ids": [], "category": "c",
                       "difficulty": "beginner", "answerable": False}
        self.golden.write_text(json.dumps(golden_case) + "\n", encoding="utf-8")
        self.rate("r1", "where is the venue", -1)
        self.rate("r2", "Something new?", -1)
        self.run_export()
        self.assertEqual([r["question"] for r in self.read_output()], ["Something new?"])

    def test_no_negative_feedback_writes_nothing(self):
        self.rate("r1", "Fine question", 1)
        code, out, _ = self.run_export()
        self.assertEqual(code, 0)
        self.assertFalse(self.output.exists())
        self.assertIn("New candidates written:        0", out)

    def test_unicode_round_trips(self):
        self.rate("r1", "Où est le lieu ? 场地在哪里", -1, "Pas d'information ✅")
        self.run_export()
        self.assertEqual(self.read_output()[0]["feedback_comments"], ["Pas d'information ✅"])

    def test_missing_database_fails_without_creating_it(self):
        code, _, err = self.run_export()
        self.assertEqual(code, 1)
        self.assertIn("not found", err)
        self.assertFalse(self.db.exists())

    def test_incompatible_schema_reports_missing_column(self):
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("CREATE TABLE feedback (id INTEGER, question TEXT, rating INTEGER)")
            connection.commit()
        code, _, err = self.run_export()
        self.assertEqual(code, 1)
        self.assertIn("missing column", err)
        self.assertIn("comment", err)

    def test_corrupt_existing_output_is_reported(self):
        self.rate("r1", "Where is the venue?", -1)
        self.output.parent.mkdir(parents=True)
        self.output.write_text("not json\n", encoding="utf-8")
        code, _, err = self.run_export()
        self.assertEqual(code, 1)
        self.assertIn("line 1", err)


if __name__ == "__main__":
    unittest.main()
