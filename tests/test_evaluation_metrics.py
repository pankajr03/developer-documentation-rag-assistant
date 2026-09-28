import unittest
from types import SimpleNamespace

from evaluation.config import DEFAULT_FALLBACK_PHRASES, EvaluationThresholds
from evaluation.judge import (
    JudgeError,
    JudgeVerdict,
    LLMJudge,
    parse_judge_output,
)
from evaluation.metrics import (
    abstention_outcome,
    check_thresholds,
    grounding_check,
    is_abstention,
    keyword_coverage,
    match_source,
    missing_metadata,
    normalize_identifier,
    retrieval_metrics,
    score_citations,
    summarize_results,
)
from evaluation.results import CaseResult
from rag.generator import FALLBACK_MESSAGE


def chunk(chunk_id, document_id="doc_a", source="Doc A.pdf", page=1):
    return {"chunk_id": chunk_id, "document_id": document_id,
            "source": source, "page": page, "text": "..."}


def citation(number, chunk_id, document_id="doc_a"):
    return {"number": number, "chunk_id": chunk_id,
            "document_id": document_id, "filename": "Doc A.pdf", "page": 1}


class SourceMatchingTests(unittest.TestCase):
    def test_matches_expected_chunk_id(self):
        self.assertEqual(
            match_source(chunk("doc_a_p1_c2"), ["doc_a_p1_c2"]),
            ("doc_a_p1_c2", "chunk_id"))

    def test_normalizes_case_and_whitespace(self):
        self.assertEqual(normalize_identifier("  Doc_A_P1_C2 "), "doc_a_p1_c2")
        self.assertEqual(
            match_source(chunk("DOC_A_p1_c2"), [" doc_a_P1_c2 "]),
            (" doc_a_P1_c2 ", "chunk_id"))

    def test_matches_document_id_when_dataset_lists_documents(self):
        self.assertEqual(
            match_source(chunk("doc_a_p1_c2"), ["doc_a"]), ("doc_a", "document_id"))

    def test_no_substring_matching(self):
        self.assertEqual(
            match_source(chunk("doc_a_p1_c2"), ["doc_a_p1_c"]), (None, None))
        self.assertEqual(
            match_source(chunk("doc_a_p1_c12"), ["doc_a_p1_c1"]), (None, None))

    def test_title_used_only_without_stable_ids(self):
        with_ids = chunk("doc_a_p1_c2")
        self.assertEqual(
            match_source(with_ids, ["other"], ["Doc A.pdf"]), (None, None))
        no_ids = {"source": "Doc A.pdf", "text": "..."}
        self.assertEqual(
            match_source(no_ids, [], [" doc a.PDF "]), (" doc a.PDF ", "title"))

    def test_reports_missing_metadata(self):
        self.assertEqual(
            missing_metadata({"source": "a.md", "chunk_id": "a_c0"}),
            ["document_id", "page"])


class RetrievalMetricTests(unittest.TestCase):
    def test_source_hit_when_expected_source_retrieved(self):
        metrics = retrieval_metrics([False, True, False], top_k=3)
        self.assertTrue(metrics["source_hit"])

    def test_source_miss(self):
        metrics = retrieval_metrics([False, False, False, False, False], top_k=5)
        self.assertFalse(metrics["source_hit"])
        self.assertEqual(metrics["reciprocal_rank"], 0.0)
        self.assertIsNone(metrics["first_relevant_rank"])
        self.assertFalse(metrics["hit_at_5"])

    def test_hit_at_k(self):
        metrics = retrieval_metrics([False, True, False, False, False], top_k=5)
        self.assertEqual(
            (metrics["hit_at_1"], metrics["hit_at_3"], metrics["hit_at_5"]),
            (False, True, True))
        metrics = retrieval_metrics([False, False, False, True, False], top_k=5)
        self.assertEqual(
            (metrics["hit_at_1"], metrics["hit_at_3"], metrics["hit_at_5"]),
            (False, False, True))

    def test_hit_at_k_is_none_when_fewer_results_requested(self):
        metrics = retrieval_metrics([True, False, False], top_k=3)
        self.assertTrue(metrics["hit_at_3"])
        self.assertIsNone(metrics["hit_at_5"])

    def test_reciprocal_rank(self):
        self.assertEqual(retrieval_metrics([True], 5)["reciprocal_rank"], 1.0)
        self.assertEqual(
            retrieval_metrics([False, True], 5)["reciprocal_rank"], 0.5)
        self.assertAlmostEqual(
            retrieval_metrics([False, False, False, True], 5)["reciprocal_rank"],
            0.25)


class KeywordCoverageTests(unittest.TestCase):
    def test_coverage_with_phrases_case_and_whitespace(self):
        coverage, matched, missing = keyword_coverage(
            "Use the  WITH statement and Open() the file.",
            ["with", " open ", "close", "with  statement"])
        self.assertEqual(matched, ["with", "open", "with  statement"])
        self.assertEqual(missing, ["close"])
        self.assertEqual(coverage, 0.75)

    def test_whole_words_only(self):
        coverage, _, missing = keyword_coverage("Do it without errors.", ["with"])
        self.assertEqual(coverage, 0.0)
        self.assertEqual(missing, ["with"])

    def test_no_expected_keywords_returns_none(self):
        self.assertEqual(keyword_coverage("Anything.", []), (None, [], []))


class CitationTests(unittest.TestCase):
    RETRIEVED = [chunk("doc_a_p1_c1"), chunk("doc_a_p1_c2")]

    def score(self, citations, unverified=(), answerable=True):
        return score_citations(
            citations, list(unverified), self.RETRIEVED,
            ["doc_a_p1_c2"], [], answerable)

    def test_citation_precision(self):
        scored = self.score(
            [citation(1, "doc_a_p1_c1"), citation(3, "doc_z_p9_c9")],
            unverified=[7])
        self.assertEqual(scored["citation_count"], 3)
        self.assertEqual(scored["citations_matching_retrieved"], 1)
        self.assertAlmostEqual(scored["citation_precision"], 1 / 3)
        self.assertEqual(len(scored["unverifiable_citations"]), 2)

    def test_no_citations_gives_no_precision(self):
        scored = self.score([])
        self.assertIsNone(scored["citation_precision"])
        self.assertFalse(scored["expected_source_citation_hit"])

    def test_expected_source_citation_hit(self):
        self.assertTrue(self.score(
            [citation(2, "doc_a_p1_c2")])["expected_source_citation_hit"])
        self.assertFalse(self.score(
            [citation(1, "doc_a_p1_c1")])["expected_source_citation_hit"])

    def test_expected_hit_is_none_for_unanswerable(self):
        self.assertIsNone(self.score(
            [citation(1, "doc_a_p1_c1")], answerable=False
        )["expected_source_citation_hit"])

    def test_citation_without_chunk_id_is_unverifiable(self):
        scored = self.score([{"number": 1, "filename": "Doc A.pdf"}])
        self.assertEqual(scored["citation_precision"], 0.0)
        self.assertIn("no chunk_id", scored["unverifiable_citations"][0])


class AbstentionTests(unittest.TestCase):
    def test_fallback_detected_despite_case_and_punctuation(self):
        self.assertTrue(is_abstention(FALLBACK_MESSAGE, DEFAULT_FALLBACK_PHRASES))
        self.assertTrue(is_abstention(
            "i COULD not find this information in the uploaded documents",
            DEFAULT_FALLBACK_PHRASES))
        self.assertTrue(is_abstention(
            "I couldn’t find that information in the indexed documentation!",
            DEFAULT_FALLBACK_PHRASES))
        self.assertFalse(is_abstention(
            "Report 30 minutes early [Source 1].", DEFAULT_FALLBACK_PHRASES))

    def test_correct_abstention_for_unanswerable(self):
        self.assertEqual(abstention_outcome(False, True, 0), (True, None))

    def test_failed_abstention_for_unanswerable(self):
        self.assertEqual(abstention_outcome(False, False, 2), (False, None))
        # Falling back while still citing chunks presents them as evidence.
        self.assertEqual(abstention_outcome(False, True, 1), (False, None))

    def test_incorrect_abstention_for_answerable(self):
        self.assertEqual(abstention_outcome(True, True, 0), (None, True))
        self.assertEqual(abstention_outcome(True, False, 1), (None, False))


class GroundingTests(unittest.TestCase):
    def test_grounded_answer(self):
        cited = score_citations(
            [citation(1, "doc_a_p1_c1")], [], [chunk("doc_a_p1_c1")],
            ["doc_a_p1_c1"], [], True)["returned_citations"]
        check = grounding_check(1, cited, False, True, True)
        self.assertTrue(check.grounded)
        self.assertFalse(check.answered_without_useful_evidence)

    def test_answer_without_citations_is_not_grounded(self):
        check = grounding_check(3, [], False, True, True)
        self.assertFalse(check.grounded)
        self.assertIsNone(check.citations_refer_to_retrieved)

    def test_answer_for_unanswerable_question_flagged(self):
        check = grounding_check(3, [], False, False, None)
        self.assertTrue(check.answered_without_useful_evidence)

    def test_fallback_is_not_scored_as_grounded(self):
        check = grounding_check(3, [], True, False, None)
        self.assertIsNone(check.grounded)
        self.assertTrue(check.used_fallback)


class SummaryTests(unittest.TestCase):
    def result(self, case_id, answerable=True, **values):
        return CaseResult(case_id=case_id, question="q", category="c",
                          difficulty="beginner", answerable=answerable, **values)

    def test_missing_measurements_are_none_not_zero(self):
        summary = summarize_results([
            self.result("a", source_hit=True, reciprocal_rank=1.0),
            self.result("b", answerable=False),
        ])
        self.assertEqual(summary.source_hit_rate, 1.0)
        self.assertIsNone(summary.average_keyword_coverage)
        self.assertIsNone(summary.correct_abstention_rate)
        self.assertIsNone(summary.citation_precision)
        self.assertIsNone(summary.incorrect_abstention_count)

    def test_rates_and_micro_citation_precision(self):
        summary = summarize_results([
            self.result("a", source_hit=True, reciprocal_rank=0.5,
                        citation_count=1, citations_matching_retrieved=1,
                        incorrect_abstention=False),
            self.result("b", source_hit=False, reciprocal_rank=0.0,
                        citation_count=3, citations_matching_retrieved=1,
                        incorrect_abstention=True),
            self.result("c", answerable=False, correct_abstention=True),
            self.result("d", error="boom"),
        ])
        self.assertEqual(summary.source_hit_rate, 0.5)
        self.assertEqual(summary.mrr, 0.25)
        self.assertEqual(summary.citation_precision, 0.5)
        self.assertEqual(summary.correct_abstention_rate, 1.0)
        self.assertEqual(summary.incorrect_abstention_count, 1)
        self.assertEqual(summary.failed_cases, 1)
        self.assertEqual(summary.completed_cases, 3)

    def test_threshold_checks(self):
        summary = summarize_results([
            self.result("a", source_hit=True, reciprocal_rank=0.5)])
        checks = check_thresholds(summary, EvaluationThresholds(
            min_source_hit_rate=0.8, min_mrr=0.6, min_keyword_coverage=0.5))
        by_metric = {check.metric: check for check in checks}
        self.assertTrue(by_metric["source_hit_rate"].passed)
        self.assertFalse(by_metric["mrr"].passed)
        # Unmeasured metrics cannot confirm the gate, so they fail it.
        self.assertFalse(by_metric["average_keyword_coverage"].passed)
        self.assertNotIn("correct_abstention_rate", by_metric)


class JudgeTests(unittest.TestCase):
    VALID = ('{"correctness": 4, "grounding": 3, "relevance": 4, '
             '"unsupported_claims": false, "reason": "Matches."}')

    def test_valid_response(self):
        verdict = parse_judge_output(self.VALID)
        self.assertEqual(verdict.correctness, 4)
        self.assertFalse(verdict.unsupported_claims)

    def test_rejects_out_of_range_missing_and_extra_fields(self):
        for text in (
            self.VALID.replace('"correctness": 4', '"correctness": 5'),
            '{"correctness": 4}',
            self.VALID.replace('}', ', "extra": 1}'),
            "not json",
        ):
            with self.subTest(text=text), self.assertRaises(JudgeError):
                parse_judge_output(text)

    def test_judge_uses_structured_output_and_low_temperature(self):
        calls = []

        class FakeResponses:
            def parse(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(output_parsed=JudgeVerdict(
                    correctness=3, grounding=4, relevance=4,
                    unsupported_claims=False, reason="ok"))

        client = SimpleNamespace(responses=FakeResponses())
        scores = LLMJudge(model="judge-model", client=client)(
            "q", "reference", "answer", [{"number": 1, "content": "ctx"}])
        self.assertEqual(scores.correctness, 3)
        self.assertEqual(calls[0]["text_format"], JudgeVerdict)
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertEqual(calls[0]["model"], "judge-model")

    def test_retries_without_temperature_when_model_rejects_it(self):
        calls = []

        class FakeResponses:
            def parse(self, **kwargs):
                calls.append(kwargs)
                if "temperature" in kwargs:
                    raise ValueError("Unsupported parameter: 'temperature'")
                return SimpleNamespace(output_parsed=None, output_text=JudgeTests.VALID)

        judge = LLMJudge(client=SimpleNamespace(responses=FakeResponses()))
        self.assertEqual(judge("q", "r", "a", []).grounding, 3)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(judge.temperature)

    def test_request_failure_raises_judge_error(self):
        class FakeResponses:
            def parse(self, **kwargs):
                raise ConnectionError("network down")

        with self.assertRaises(JudgeError):
            LLMJudge(client=SimpleNamespace(responses=FakeResponses()))(
                "q", "r", "a", [])


if __name__ == "__main__":
    unittest.main()
