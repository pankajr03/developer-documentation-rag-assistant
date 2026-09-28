import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from scripts import run_rag_evaluation as cli
from tests.evaluation_fakes import FakeGenerator, FakeRetriever, write_dataset


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = write_dataset(self.root / "golden.jsonl")

    def main(self, *args):
        argv = ["--dataset", str(self.dataset), "--top-k", "3",
                "--output-dir", str(self.root / "reports"), *args]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(argv, retrieve=FakeRetriever(),
                            generate=FakeGenerator())
        return code, output.getvalue()

    def test_threshold_success_exit_code(self):
        code, output = self.main(
            "--fail-below-source-hit", "0.8", "--fail-below-mrr", "0.6",
            "--fail-below-keyword-coverage", "0.5",
            "--fail-below-abstention", "0.8")
        self.assertEqual(code, cli.EXIT_PASS)
        self.assertIn("Result: PASS", output)
        self.assertIn("MRR: 0.667", output)

    def test_threshold_failure_exit_code_keeps_reports(self):
        code, output = self.main("--fail-below-mrr", "0.9")
        self.assertEqual(code, cli.EXIT_FAIL)
        self.assertIn("Result: FAIL", output)
        self.assertIn("mrr >= 0.9", output)
        run_dirs = list((self.root / "reports").iterdir())
        self.assertEqual(len(run_dirs), 1)
        self.assertTrue((run_dirs[0] / "summary.json").is_file())

    def test_recommended_thresholds_can_be_overridden(self):
        args = cli.build_parser().parse_args(
            ["--recommended-thresholds", "--fail-below-mrr", "0.1"])
        thresholds = cli.config_from_args(args).thresholds
        self.assertEqual(thresholds.min_mrr, 0.1)
        self.assertEqual(thresholds.min_source_hit_rate, 0.8)

    def test_invalid_configuration_exit_code(self):
        code, output = self.main("--top-k", "0")
        self.assertEqual(code, cli.EXIT_SETUP_ERROR)
        self.assertIn("Invalid configuration", output)

    def test_unknown_case_id_exit_code(self):
        code, output = self.main("--case-id", "eval_999")
        self.assertEqual(code, cli.EXIT_SETUP_ERROR)
        self.assertIn("eval_999", output)

    def test_missing_dataset_exit_code(self):
        self.dataset = self.root / "missing.jsonl"
        code, _ = self.main()
        self.assertEqual(code, cli.EXIT_SETUP_ERROR)

    def test_skip_generation_and_filters(self):
        code, output = self.main("--skip-generation", "--category", "rules")
        self.assertEqual(code, cli.EXIT_PASS)
        self.assertIn("Cases evaluated: 2", output)
        self.assertIn("Keyword coverage: n/a", output)


if __name__ == "__main__":
    unittest.main()
