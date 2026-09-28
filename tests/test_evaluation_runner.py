import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from evaluation.config import EvaluationConfig
from evaluation.dataset import EvaluationDatasetError
from evaluation.runner import EvaluationSetupError, run_evaluation
from tests.evaluation_fakes import CASES, FakeGenerator, FakeRetriever, write_dataset


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = write_dataset(self.root / "golden.jsonl")
        self.retriever = FakeRetriever()
        self.generator = FakeGenerator()

    def config(self, **values):
        values.setdefault("output_directory", str(self.root / "reports"))
        values.setdefault("top_k", 3)
        return EvaluationConfig(**values)

    def run_eval(self, **values):
        return run_evaluation(
            self.dataset, self.config(**values),
            retrieve=self.retriever, generate=self.generator)


class RunnerTests(RunnerTestCase):
    def test_scores_every_case(self):
        run = self.run_eval()
        by_id = {r.case_id: r for r in run.results}

        first = by_id["eval_001"]
        self.assertTrue(first.source_hit)
        self.assertTrue(first.hit_at_1)
        self.assertEqual(first.reciprocal_rank, 1.0)
        self.assertEqual(first.keyword_coverage, 1.0)
        self.assertEqual(first.citation_precision, 1.0)
        self.assertTrue(first.expected_source_citation_hit)
        self.assertFalse(first.incorrect_abstention)
        self.assertEqual(first.failure_reasons, [])

        second = by_id["eval_002"]
        self.assertFalse(second.hit_at_1)
        self.assertTrue(second.hit_at_3)
        self.assertIsNone(second.hit_at_5)  # only 3 results requested
        self.assertAlmostEqual(second.reciprocal_rank, 1 / 3)
        self.assertEqual(second.missing_keywords, ["RSFI"])
        self.assertEqual(second.retrieved_sources[2].matched_on, "chunk_id")

        unanswerable = by_id["eval_003"]
        self.assertIsNone(unanswerable.source_hit)
        self.assertIsNone(unanswerable.reciprocal_rank)
        self.assertEqual(len(unanswerable.retrieved_sources), 3)
        self.assertTrue(unanswerable.abstained)
        self.assertTrue(unanswerable.correct_abstention)

        metrics = run.summary["metrics"]
        self.assertEqual(metrics["total_cases"], 3)
        self.assertEqual(metrics["answerable_cases"], 2)
        self.assertEqual(metrics["source_hit_rate"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], round((1 + 1 / 3) / 2, 4))
        self.assertEqual(metrics["correct_abstention_rate"], 1.0)
        self.assertTrue(run.passed)

    def test_only_the_question_reaches_the_pipeline(self):
        self.run_eval()
        golden_values = []
        for case in CASES:
            golden_values += [case["expected_answer"], *case["expected_keywords"],
                              *case["expected_source_ids"]]
        for args, kwargs in self.retriever.calls + self.generator.calls:
            self.assertEqual(set(kwargs), {"top_k"})
            self.assertIn(args[0], [case["question"] for case in CASES])
            sent = json.dumps([args[0], kwargs])
            for value in golden_values:
                self.assertNotIn(value, sent)

    def test_failed_abstention_on_unanswerable_case(self):
        self.generator = FakeGenerator(answers={
            **FakeGenerator().answers,
            "Where is the venue?": "It is at the city rink [Source 1].",
        })
        result = {r.case_id: r for r in self.run_eval().results}["eval_003"]
        self.assertFalse(result.abstained)
        self.assertFalse(result.correct_abstention)
        self.assertTrue(result.grounding.answered_without_useful_evidence)
        self.assertIn("failed_abstention", result.failure_reasons)

    def test_incorrect_abstention_on_answerable_case(self):
        self.generator = FakeGenerator(answers={
            **FakeGenerator().answers,
            "What is the reporting time?":
                "I could not find this information in the uploaded documents.",
        })
        run = self.run_eval()
        result = {r.case_id: r for r in run.results}["eval_001"]
        self.assertTrue(result.incorrect_abstention)
        self.assertIn("incorrect_abstention", result.failure_reasons)
        self.assertEqual(run.summary["metrics"]["incorrect_abstention_count"], 1)

    def test_one_failing_case_does_not_stop_the_run(self):
        self.retriever = FakeRetriever(fail_on=("What is the reporting time?",))
        self.generator = FakeGenerator(fail_on=("How long is one lap?",))
        run = self.run_eval()
        by_id = {r.case_id: r for r in run.results}
        self.assertEqual(len(run.results), 3)
        self.assertEqual(by_id["eval_001"].error_stage, "retrieval")
        self.assertIn("retrieval exploded", by_id["eval_001"].error)
        self.assertEqual(by_id["eval_002"].error_stage, "generation")
        # Retrieval metrics survive a generation failure.
        self.assertTrue(by_id["eval_002"].source_hit)
        self.assertIsNone(by_id["eval_003"].error)
        self.assertEqual(run.summary["errors"]["case_ids"], ["eval_001", "eval_002"])
        self.assertFalse(run.passed)  # default policy: no case errors allowed

    def test_case_errors_allowed_by_policy(self):
        self.retriever = FakeRetriever(fail_on=("Where is the venue?",))
        self.assertTrue(self.run_eval(max_case_errors=1).passed)

    def test_skip_generation_gives_retrieval_only(self):
        run = self.run_eval(generate_answers=False)
        self.assertEqual(self.generator.calls, [])
        first = run.results[0]
        self.assertTrue(first.source_hit)
        self.assertIsNone(first.generated_answer)
        self.assertIsNone(first.keyword_coverage)
        self.assertIsNone(run.summary["metrics"]["correct_abstention_rate"])
        self.assertIsNone(run.summary["models"]["generation_model"])

    def test_judge_failure_is_recorded_without_losing_the_case(self):
        def broken_judge(*args):
            raise RuntimeError("judge down")

        run = run_evaluation(
            self.dataset, self.config(use_llm_judge=True),
            retrieve=self.retriever, generate=self.generator, judge=broken_judge)
        self.assertTrue(all(r.judge_error for r in run.results))
        self.assertTrue(all(r.error is None for r in run.results))
        self.assertEqual(run.summary["metrics"]["judge_failures"], 3)
        self.assertEqual(run.summary["metrics"]["source_hit_rate"], 1.0)

    def test_groups_by_category_difficulty_and_answerability(self):
        summary = self.run_eval().summary
        self.assertEqual(set(summary["by_category"]), {"format", "rules"})
        self.assertEqual(summary["by_category"]["rules"]["total_cases"], 2)
        self.assertEqual(set(summary["by_difficulty"]),
                         {"beginner", "intermediate", "advanced"})
        self.assertEqual(set(summary["by_answerability"]),
                         {"answerable", "unanswerable"})


class FilterTests(RunnerTestCase):
    def selected(self, **values):
        return [r.case_id for r in self.run_eval(**values).results]

    def test_filter_by_category(self):
        self.assertEqual(self.selected(category="rules"), ["eval_001", "eval_003"])

    def test_filter_by_difficulty(self):
        self.assertEqual(self.selected(difficulty="intermediate"), ["eval_002"])

    def test_filter_by_case_id(self):
        self.assertEqual(self.selected(case_ids=["eval_003", "eval_001"]),
                         ["eval_001", "eval_003"])

    def test_max_cases(self):
        self.assertEqual(self.selected(max_cases=2), ["eval_001", "eval_002"])

    def test_unknown_case_id_is_a_clear_error(self):
        with self.assertRaisesRegex(EvaluationSetupError, "eval_999"):
            self.run_eval(case_ids=["eval_999"])

    def test_filter_matching_nothing_is_a_clear_error(self):
        with self.assertRaisesRegex(EvaluationSetupError, "No cases match"):
            self.run_eval(category="missing")
        self.assertFalse((self.root / "reports").exists())

    def test_empty_dataset_fails_clearly(self):
        self.dataset.write_text("\n", encoding="utf-8")
        with self.assertRaisesRegex(EvaluationDatasetError, "no cases"):
            self.run_eval()

    def test_invalid_config(self):
        for values in ({"top_k": 0}, {"difficulty": "expert"},
                       {"use_llm_judge": True, "generate_answers": False},
                       {"max_cases": 0}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                EvaluationConfig(**values)

    def test_vector_store_check_failure_stops_before_reports(self):
        def unavailable():
            raise EvaluationSetupError("vector database unavailable")

        with self.assertRaises(EvaluationSetupError):
            run_evaluation(self.dataset, self.config(), retrieve=self.retriever,
                           generate=self.generator,
                           vector_store_check=unavailable)
        self.assertFalse((self.root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
