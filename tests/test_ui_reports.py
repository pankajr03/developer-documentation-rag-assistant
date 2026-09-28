"""Tests for the read-only evaluation report access in ui.reports."""

import json
import tempfile
import unittest
from pathlib import Path

from ui import reports
from ui.reports import ReportError


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.baseline = root / "reports" / "baseline"
        self.runs = root / "reports" / "evaluation"
        self.secret = root / ".env"
        self.secret.write_text("OPENAI_API_KEY=sk-test", encoding="utf-8")

    def make_report(self, folder, summary=None, failures=None):
        folder.mkdir(parents=True)
        if summary is not False:
            (folder / "summary.json").write_text(
                json.dumps(summary or {"metrics": {"total_cases": 2, "mrr": 0.5}}),
                encoding="utf-8")
        if failures is not None:
            (folder / "failures.jsonl").write_text(failures, encoding="utf-8")
        return folder

    def refs(self):
        return reports.list_reports(self.baseline, self.runs)

    def test_lists_reports_newest_first_with_baseline_last(self):
        self.make_report(self.baseline)
        self.make_report(self.runs / "2026-09-27_100000")
        self.make_report(self.runs / "2026-09-28_120000")
        self.make_report(self.runs / "no_summary", summary=False)
        self.assertEqual([r.report_id for r in self.refs()], [
            "runs/2026-09-28_120000", "runs/2026-09-27_100000", "baseline"])

    def test_no_reports(self):
        self.assertEqual(self.refs(), [])

    def test_only_listed_ids_resolve(self):
        self.make_report(self.runs / "2026-09-28_120000")
        refs = self.refs()
        self.assertEqual(reports.resolve_report("runs/2026-09-28_120000", refs).path,
                         self.runs / "2026-09-28_120000")
        for bad in ("../.env", "runs/../../.env", "runs/..", "/etc/passwd",
                    str(self.secret), "baseline", ""):
            with self.subTest(bad=bad), self.assertRaises(ReportError):
                reports.resolve_report(bad, refs)

    def test_unsafe_folder_names_are_not_listed(self):
        self.make_report(self.runs / "has space")
        self.make_report(self.runs / "dots.in.name")
        self.assertEqual(self.refs(), [])

    def test_load_summary(self):
        self.make_report(self.baseline, {"metrics": {"mrr": 0.64}, "passed": True})
        summary = reports.load_summary(self.refs()[0])
        self.assertEqual(summary["metrics"]["mrr"], 0.64)

    def test_malformed_summaries(self):
        self.make_report(self.baseline)
        ref = self.refs()[0]
        for content in ("{not json", "[1, 2]", '{"no_metrics": 1}',
                        '{"metrics": "oops"}'):
            with self.subTest(content=content):
                (self.baseline / "summary.json").write_text(content, encoding="utf-8")
                with self.assertRaises(ReportError):
                    reports.load_summary(ref)
        (self.baseline / "summary.json").unlink()
        with self.assertRaisesRegex(ReportError, "no summary.json"):
            reports.load_summary(ref)

    def test_failures_are_limited_and_bad_lines_counted(self):
        lines = [json.dumps({"case_id": f"eval_{i}", "failure_reasons": ["error"],
                             "expected_answer": "SECRET"}) for i in range(5)]
        self.make_report(self.baseline,
                         failures="\n".join([lines[0], "{broken", *lines[1:]]) + "\n")
        records, bad = reports.load_failures(self.refs()[0], limit=3)
        self.assertEqual([r["case_id"] for r in records],
                         ["eval_0", "eval_1", "eval_2"])
        self.assertEqual(bad, 1)
        rows = reports.failure_rows(records)
        self.assertEqual(rows[0]["Reasons"], "error")
        self.assertNotIn("SECRET", str(rows))

    def test_missing_failures_file(self):
        self.make_report(self.baseline)
        self.assertEqual(reports.load_failures(self.refs()[0]), ([], 0))

    def test_downloads_are_allow_listed(self):
        self.make_report(self.baseline)
        ref = self.refs()[0]
        self.assertIn(b"metrics", reports.read_download(ref, "summary.json"))
        self.assertIsNone(reports.read_download(ref, "results.csv"))  # missing
        for bad in ("../../.env", "results.jsonl", ".env"):
            with self.subTest(bad=bad), self.assertRaises(ReportError):
                reports.read_download(ref, bad)


if __name__ == "__main__":
    unittest.main()
