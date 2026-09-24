import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetNotFoundError,
    EvaluationDatasetError,
    find_missing_source_ids,
    load_evaluation_dataset,
    read_evaluation_dataset,
    summarize_dataset,
)
from evaluation.schemas import EvaluationCase
from scripts import validate_evaluation_dataset as validator

ROOT = Path(__file__).resolve().parent.parent


def make_case(case_id="eval_001", **overrides):
    case = {
        "id": case_id,
        "question": "How do I open a file safely in Python?",
        "expected_answer": "Use the with statement and open().",
        "expected_keywords": ["with", "open", "close"],
        "expected_source_ids": ["python_docs_file_io"],
        "expected_source_titles": ["Reading and Writing Files"],
        "category": "file_io",
        "difficulty": "beginner",
        "answerable": True,
        "notes": "Mention automatic closing.",
    }
    case.update(overrides)
    return case


class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "dataset.jsonl"

    def write_lines(self, *lines):
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_cases(self, *cases):
        self.write_lines(*(json.dumps(c) for c in cases))


class LoadTests(DatasetTestCase):
    def test_loads_valid_dataset_as_typed_objects(self):
        self.write_cases(make_case("a"), make_case("b", category="strings"))
        cases = load_evaluation_dataset(self.path)
        self.assertEqual([c.id for c in cases], ["a", "b"])
        self.assertIsInstance(cases[0], EvaluationCase)
        self.assertEqual(cases[0].expected_source_titles, ["Reading and Writing Files"])

    def test_blank_lines_are_ignored_and_line_numbers_still_match(self):
        self.write_lines(json.dumps(make_case("a")), "", "   ", "{bad json")
        result = read_evaluation_dataset(self.path)
        self.assertEqual(len(result.cases), 1)
        self.assertEqual(result.issues[0].line, 4)

    def test_blank_lines_alone_are_fine(self):
        self.write_lines("", json.dumps(make_case("a")), "", json.dumps(make_case("b")), "")
        self.assertEqual(len(load_evaluation_dataset(self.path)), 2)

    def test_optional_fields_default_to_empty(self):
        minimal = {k: v for k, v in make_case().items()
                   if k not in ("expected_source_titles", "notes")}
        self.write_cases(minimal)
        case = load_evaluation_dataset(self.path)[0]
        self.assertEqual((case.expected_source_titles, case.tags, case.metadata, case.notes),
                         ([], [], {}, ""))

    def test_null_lists_become_empty_lists(self):
        self.write_cases(make_case(expected_keywords=None, tags=None,
                                   expected_source_titles=None, metadata=None, notes=None))
        case = load_evaluation_dataset(self.path)[0]
        self.assertEqual((case.expected_keywords, case.tags, case.expected_source_titles),
                         ([], [], []))

    def test_utf8_and_bom(self):
        case = make_case(question="Comment ouvrir un fichier ? 文件 ✅")
        self.path.write_bytes(b"\xef\xbb\xbf" + (json.dumps(case, ensure_ascii=False) + "\n").encode("utf-8"))
        self.assertEqual(load_evaluation_dataset(self.path)[0].question, case["question"])

    def test_loads_unanswerable_case(self):
        self.write_cases(make_case(
            "no_answer", question="What is the entry fee?",
            expected_answer="The documents do not contain this information.",
            expected_keywords=[], expected_source_ids=[],
            expected_source_titles=[], answerable=False))
        case = load_evaluation_dataset(self.path)[0]
        self.assertFalse(case.answerable)
        self.assertEqual(case.expected_source_ids, [])

    def test_missing_file_raises_clear_error(self):
        with self.assertRaises(DatasetNotFoundError) as context:
            load_evaluation_dataset(Path(self.tmp.name) / "nope.jsonl")
        self.assertIn("not found", str(context.exception))

    def test_rejects_malformed_json_with_line_number(self):
        self.write_lines(json.dumps(make_case("a")), '{"id": "b", "question": ')
        with self.assertRaises(EvaluationDatasetError) as context:
            load_evaluation_dataset(self.path)
        self.assertEqual(context.exception.issues[0].line, 2)
        self.assertIn("malformed JSON", str(context.exception))

    def test_rejects_non_object_line(self):
        self.write_lines("[1, 2, 3]")
        with self.assertRaises(EvaluationDatasetError):
            load_evaluation_dataset(self.path)

    def test_rejects_missing_required_field(self):
        case = make_case()
        del case["expected_answer"]
        self.write_cases(make_case("a"), case)
        with self.assertRaises(EvaluationDatasetError) as context:
            load_evaluation_dataset(self.path)
        issue = context.exception.issues[0]
        self.assertEqual(issue.line, 2)
        self.assertIn("expected_answer", issue.message)

    def test_rejects_invalid_difficulty(self):
        self.write_cases(make_case(difficulty="expert"))
        with self.assertRaises(EvaluationDatasetError) as context:
            load_evaluation_dataset(self.path)
        self.assertIn("difficulty", str(context.exception))

    def test_detects_duplicate_ids(self):
        self.write_cases(make_case("dup"), make_case("other"), make_case("dup"))
        with self.assertRaises(EvaluationDatasetError) as context:
            load_evaluation_dataset(self.path)
        issue = context.exception.issues[0]
        self.assertEqual(issue.line, 3)
        self.assertIn("duplicate id 'dup'", issue.message)
        self.assertIn("line 1", issue.message)

    def test_reports_every_invalid_line(self):
        self.write_lines("not json", json.dumps(make_case("a", difficulty="x")),
                         json.dumps(make_case("ok")))
        result = read_evaluation_dataset(self.path)
        self.assertEqual([i.line for i in result.issues], [1, 2])
        self.assertEqual([c.id for c in result.cases], ["ok"])

    def test_empty_dataset_is_an_error(self):
        self.write_lines("", "  ")
        with self.assertRaises(EvaluationDatasetError):
            load_evaluation_dataset(self.path)

    def test_invalid_utf8_is_an_error(self):
        self.path.write_bytes(b'{"id": "\xff\xfe"}\n')
        with self.assertRaises(EvaluationDatasetError) as context:
            load_evaluation_dataset(self.path)
        self.assertIn("UTF-8", str(context.exception))


class SchemaTests(unittest.TestCase):
    def validate(self, **overrides):
        return EvaluationCase.model_validate(make_case(**overrides))

    def test_empty_id_or_question_rejected(self):
        for field in ("id", "question", "category", "expected_answer"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(**{field: "   "})

    def test_answerable_needs_expected_answer_and_a_source(self):
        with self.assertRaises(ValueError):
            self.validate(expected_answer="")
        with self.assertRaises(ValueError):
            self.validate(expected_source_ids=[])

    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(expected_keyword=["typo"])

    def test_answerable_must_be_a_real_boolean(self):
        with self.assertRaises(ValueError):
            self.validate(answerable="yes")

    def test_empty_list_items_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(expected_keywords=["ok", " "])

    def test_all_three_difficulties_accepted(self):
        for level in ("beginner", "intermediate", "advanced"):
            self.assertEqual(self.validate(difficulty=level).difficulty, level)

    def test_values_are_trimmed(self):
        self.assertEqual(self.validate(id="  eval_9 ").id, "eval_9")


class SummaryTests(DatasetTestCase):
    def test_counts(self):
        self.write_cases(
            make_case("a"),
            make_case("b", category="strings", difficulty="advanced"),
            make_case("c", answerable=False, expected_source_ids=[]),
        )
        summary = summarize_dataset(load_evaluation_dataset(self.path))
        self.assertEqual(summary["total"], 3)
        self.assertEqual((summary["answerable"], summary["unanswerable"]), (2, 1))
        self.assertEqual(summary["by_category"], {"file_io": 2, "strings": 1})
        self.assertEqual(summary["by_difficulty"],
                         {"beginner": 2, "intermediate": 0, "advanced": 1})

    def test_find_missing_source_ids(self):
        self.write_cases(make_case("a", expected_source_ids=["x", "y"]),
                         make_case("b", expected_source_ids=["x"]))
        cases = load_evaluation_dataset(self.path)
        self.assertEqual(find_missing_source_ids(cases, ["x"]), {"a": ["y"]})
        self.assertEqual(find_missing_source_ids(cases, ["x", "y"]), {})


class ValidatorCliTests(DatasetTestCase):
    def run_main(self, *args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = validator.main(list(args))
        return code, buffer.getvalue()

    def test_success_exit_code_and_report(self):
        self.write_cases(make_case("a"), make_case("b", answerable=False,
                                                   expected_source_ids=[], difficulty="advanced"))
        code, output = self.run_main("--dataset", str(self.path))
        self.assertEqual(code, 0)
        for expected in (str(self.path), "Total valid cases: 2", "Answerable: 1",
                         "Unanswerable: 1", "file_io: 2", "beginner: 1",
                         "advanced: 1", "Result: PASS"):
            self.assertIn(expected, output)

    def test_failure_exit_code_on_invalid_record(self):
        self.write_lines(json.dumps(make_case("a")), "{oops")
        code, output = self.run_main("--dataset", str(self.path))
        self.assertEqual(code, 1)
        self.assertIn("line 2", output)
        self.assertIn("Result: FAIL", output)

    def test_failure_exit_code_on_duplicate_ids(self):
        self.write_cases(make_case("a"), make_case("a"))
        code, output = self.run_main("--dataset", str(self.path))
        self.assertEqual(code, 1)
        self.assertIn("duplicate id", output)

    def test_failure_exit_code_on_missing_file(self):
        code, output = self.run_main("--dataset", str(Path(self.tmp.name) / "none.jsonl"))
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_check_index_reports_missing_sources(self):
        self.write_cases(make_case("a", expected_source_ids=["present", "gone"]))
        original = validator._indexed_chunk_ids
        validator._indexed_chunk_ids = lambda: {"present"}
        self.addCleanup(setattr, validator, "_indexed_chunk_ids", original)
        code, output = self.run_main("--dataset", str(self.path), "--check-index")
        self.assertEqual(code, 1)
        self.assertIn("gone", output)

    def test_check_index_passes_when_all_ids_exist(self):
        self.write_cases(make_case("a", expected_source_ids=["present"]))
        original = validator._indexed_chunk_ids
        validator._indexed_chunk_ids = lambda: {"present"}
        self.addCleanup(setattr, validator, "_indexed_chunk_ids", original)
        code, output = self.run_main("--dataset", str(self.path), "--check-index")
        self.assertEqual(code, 0)
        self.assertIn("Result: PASS", output)

    def test_script_process_exit_codes(self):
        self.write_cases(make_case("a"))
        script = str(ROOT / "scripts" / "validate_evaluation_dataset.py")
        ok = subprocess.run([sys.executable, script, "--dataset", str(self.path)],
                            capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.write_lines("{bad")
        bad = subprocess.run([sys.executable, script, "--dataset", str(self.path)],
                             capture_output=True, text=True, cwd=self.tmp.name)
        self.assertNotEqual(bad.returncode, 0)


class StarterDatasetTests(unittest.TestCase):
    """The committed golden dataset must always be valid (read-only checks)."""

    def test_golden_dataset_is_valid_and_meets_the_minimums(self):
        cases = load_evaluation_dataset(DEFAULT_DATASET_PATH)
        summary = summarize_dataset(cases)
        self.assertGreaterEqual(summary["total"], 15)
        self.assertGreaterEqual(summary["answerable"], 12)
        self.assertGreaterEqual(summary["unanswerable"], 3)
        for case in cases:
            if not case.answerable:
                self.assertEqual(case.expected_source_ids, [])


if __name__ == "__main__":
    unittest.main()
