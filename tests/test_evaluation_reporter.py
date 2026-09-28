import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from evaluation.config import EvaluationConfig
from evaluation.reporter import CSV_FIELDS, create_run_directory
from evaluation.runner import run_evaluation
from tests.evaluation_fakes import FakeGenerator, FakeRetriever, write_dataset

NOW = datetime(2026, 9, 28, 12, 0, 0)


class ReporterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = write_dataset(self.root / "golden.jsonl")
        self.reports = self.root / "reports"

    def run_eval(self, **values):
        values.setdefault("output_directory", str(self.reports))
        return run_evaluation(
            self.dataset, EvaluationConfig(top_k=3, **values),
            retrieve=FakeRetriever(),
            generate=FakeGenerator(answers={
                "What is the reporting time?": "Report early [Source 1].",
                "How long is one lap?": "One lap is 200m, RSFI [Source 3].",
                "Where is the venue?": "At the rink [Source 2].",
            }),
            now=NOW)

    def test_creates_all_report_files(self):
        run = self.run_eval()
        self.assertEqual(run.report_dir, self.reports / "2026-09-28_120000")
        names = sorted(path.name for path in run.report_dir.iterdir())
        self.assertEqual(names, ["failures.jsonl", "results.csv",
                                 "results.jsonl", "summary.json"])

        summary = json.loads((run.report_dir / "summary.json").read_text("utf-8"))
        for key in ("timestamp", "dataset", "config", "models", "vector_store",
                    "metrics", "by_category", "by_difficulty", "errors",
                    "environment", "thresholds"):
            self.assertIn(key, summary)
        self.assertEqual(summary["dataset"]["case_count"], 3)
        self.assertEqual(summary["config"]["top_k"], 3)
        self.assertEqual(summary["vector_store"]["collection"], "developer_docs")
        self.assertEqual(summary["models"]["embedding_model"],
                         "text-embedding-3-small")

        lines = (run.report_dir / "results.jsonl").read_text("utf-8").splitlines()
        self.assertEqual([json.loads(line)["case_id"] for line in lines],
                         ["eval_001", "eval_002", "eval_003"])

        with (run.report_dir / "results.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(tuple(rows[0]), CSV_FIELDS)

        failures = [json.loads(line) for line in
                    (run.report_dir / "failures.jsonl").read_text("utf-8").splitlines()]
        reasons = {f["case_id"]: f["failure_reasons"] for f in failures}
        self.assertIn("low_keyword_coverage", reasons["eval_001"])
        self.assertIn("failed_abstention", reasons["eval_003"])
        self.assertNotIn("eval_002", reasons)

    def test_reports_contain_no_secrets(self):
        run = self.run_eval()
        text = "".join(p.read_text("utf-8") for p in run.report_dir.iterdir())
        self.assertNotIn("OPENAI_API_KEY", text)
        self.assertNotIn("sk-", text)

    def test_never_overwrites_an_existing_run(self):
        first = self.run_eval()
        marker = first.report_dir / "summary.json"
        original = marker.read_text("utf-8")
        second = self.run_eval()
        self.assertNotEqual(first.report_dir, second.report_dir)
        self.assertEqual(second.report_dir.name, "2026-09-28_120000_2")
        self.assertEqual(marker.read_text("utf-8"), original)

    def test_explicit_run_directory_must_be_new_or_empty(self):
        target = self.root / "baseline"
        run = self.run_eval(run_directory=str(target))
        self.assertEqual(run.report_dir, target)
        with self.assertRaises(FileExistsError):
            self.run_eval(run_directory=str(target))

    def test_create_run_directory_uses_timestamp(self):
        config = EvaluationConfig(output_directory=str(self.reports))
        path = create_run_directory(config, NOW)
        self.assertTrue(path.is_dir())
        self.assertEqual(path.name, "2026-09-28_120000")


if __name__ == "__main__":
    unittest.main()
