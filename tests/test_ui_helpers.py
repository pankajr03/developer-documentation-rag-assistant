"""Tests for the Streamlit-free UI helpers: state, validation, formatting."""

import unittest

from ui import state
from ui.config import DEFAULT_TOP_K, MAX_EXCERPT_CHARS, MAX_QUESTION_CHARS
from ui.formatting import (
    dedupe_sources,
    display_filename,
    escape_markdown,
    format_rate,
    format_score,
    format_source,
    format_sources,
    safe_answer_markdown,
    truncate_excerpt,
)
from ui.validation import (
    QuestionValidationError,
    sanitize_filename,
    validate_question,
    validate_upload,
)

SOURCE = {"number": 1, "filename": "Guide.pdf", "page": 3, "section": "Setup",
          "chunk_id": "guide_p3_c2", "document_id": "guide",
          "content": "Install the package.", "distance": 0.4213}


class StateTests(unittest.TestCase):
    def setUp(self):
        self.session = {}
        state.init_state(self.session)

    def test_init_sets_defaults_without_overwriting(self):
        self.assertEqual(self.session["messages"], [])
        self.assertEqual(self.session["top_k"], DEFAULT_TOP_K)
        session_id = self.session["session_id"]
        self.session["top_k"] = 7
        state.init_state(self.session)
        self.assertEqual(self.session["top_k"], 7)
        self.assertEqual(self.session["session_id"], session_id)

    def test_add_user_message(self):
        message = state.add_user_message(self.session, "How do I install it?")
        self.assertEqual(self.session["messages"], [message])
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"], "How do I install it?")

    def test_assistant_message_keeps_the_generator_response_id(self):
        message = state.add_assistant_message(self.session, {
            "content": "Answer [Source 1].", "response_id": "resp-1",
            "sources": [format_source(SOURCE, True)]})
        self.assertEqual(message["response_id"], "resp-1")
        self.assertEqual(message["kind"], "answer")

    def test_citations_survive_reruns(self):
        state.add_assistant_message(self.session, {
            "content": "A", "response_id": "resp-1",
            "sources": [format_source(SOURCE, True)]})
        stored = [dict(m) for m in self.session["messages"]]
        # A rerun calls init_state again and redisplays the stored messages.
        state.init_state(self.session)
        state.mark_feedback_submitted(self.session, "resp-1", 1)
        self.assertEqual(self.session["messages"], stored)
        self.assertTrue(self.session["messages"][0]["sources"][0]["cited"])

    def test_duplicate_feedback_is_refused(self):
        self.assertTrue(state.mark_feedback_submitted(self.session, "resp-1", 1))
        self.assertFalse(state.mark_feedback_submitted(self.session, "resp-1", -1))
        self.assertEqual(state.feedback_rating(self.session, "resp-1"), 1)

    def test_failed_feedback_can_be_retried(self):
        state.mark_feedback_failed(self.session, "resp-1", "db down")
        self.assertEqual(state.feedback_error(self.session, "resp-1"), "db down")
        self.assertIsNone(state.feedback_rating(self.session, "resp-1"))
        self.assertTrue(state.mark_feedback_submitted(self.session, "resp-1", -1))

    def test_error_message_has_no_response_id(self):
        message = state.add_error_message(self.session, "q", "Something failed.")
        self.assertIsNone(message["response_id"])
        self.assertEqual(message["kind"], "error")

    def test_clear_conversation_keeps_settings(self):
        state.add_user_message(self.session, "q")
        state.mark_feedback_submitted(self.session, "resp-1", 1)
        self.session["top_k"] = 6
        state.clear_conversation(self.session)
        self.assertEqual(self.session["messages"], [])
        self.assertEqual(self.session["feedback_status"], {})
        self.assertEqual(self.session["top_k"], 6)


class QuestionValidationTests(unittest.TestCase):
    def test_empty_and_whitespace_questions_are_rejected(self):
        for question in (None, "", "   \n\t "):
            with self.subTest(question=question), \
                    self.assertRaisesRegex(QuestionValidationError, "enter a question"):
                validate_question(question)

    def test_question_is_trimmed(self):
        self.assertEqual(validate_question("  What is X?  "), "What is X?")

    def test_question_length_limit(self):
        self.assertEqual(len(validate_question("a" * MAX_QUESTION_CHARS)),
                         MAX_QUESTION_CHARS)
        with self.assertRaisesRegex(QuestionValidationError, "under"):
            validate_question("a" * (MAX_QUESTION_CHARS + 1))
        with self.assertRaises(QuestionValidationError):
            validate_question("abcdef", max_chars=5)


class UploadValidationTests(unittest.TestCase):
    def test_allowed_extensions(self):
        for name in ("guide.pdf", "notes.TXT", "README.md"):
            with self.subTest(name=name):
                self.assertTrue(validate_upload(name, 100).ok)

    def test_rejected_extensions(self):
        for name in ("script.py", "page.html", "archive.zip", "noextension",
                     "report.pdf.exe", "doc.docx"):
            with self.subTest(name=name):
                check = validate_upload(name, 100)
                self.assertFalse(check.ok)
                self.assertIn("Unsupported file type", check.message)

    def test_empty_file_rejected(self):
        self.assertIn("empty", validate_upload("a.txt", 0).message)

    def test_size_limit(self):
        self.assertTrue(validate_upload("a.txt", 1024, max_bytes=1024).ok)
        check = validate_upload("a.txt", 1025, max_bytes=1024)
        self.assertFalse(check.ok)
        self.assertIn("limit", check.message)

    def test_filename_sanitization(self):
        cases = {
            "../../etc/passwd.txt": "passwd.txt",
            r"C:\Users\me\secret\notes.md": "notes.md",
            "Speed Schedule (1).pdf": "Speed Schedule (1).pdf",
            "bad<name>|x.pdf": "bad_name__x.pdf",
            "tab\tnew\nline.txt": "tabnewline.txt",
            "..": "document",
            "": "document",
            "   .hidden.md": "hidden.md",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_filename(raw), expected)

    def test_long_filename_keeps_extension(self):
        name = sanitize_filename("x" * 500 + ".pdf")
        self.assertLessEqual(len(name), 120)
        self.assertTrue(name.endswith(".pdf"))

    def test_upload_reports_sanitized_name(self):
        self.assertEqual(validate_upload("../x/guide.md", 5).display_name, "guide.md")


class FormattingTests(unittest.TestCase):
    def test_source_metadata_formatting(self):
        formatted = format_source(SOURCE, cited=True)
        self.assertEqual(formatted["label"], "[1] Guide.pdf — page 3")
        self.assertEqual(dict(formatted["fields"]), {
            "Document": "Guide.pdf", "Page": "3", "Section": "Setup",
            "Chunk ID": "guide_p3_c2", "Document ID": "guide",
            "Distance (lower is closer)": "0.421"})
        self.assertTrue(formatted["cited"])
        self.assertEqual(formatted["excerpt"], "Install the package.")

    def test_missing_fields_are_omitted_not_none(self):
        formatted = format_source(
            {"number": 2, "filename": "notes.md", "page": None, "section": None,
             "chunk_id": None, "distance": None, "content": ""}, cited=False)
        self.assertEqual(formatted["fields"], [("Document", "notes.md")])
        self.assertEqual(formatted["label"], "[2] notes.md")
        self.assertNotIn("None", str(formatted))

    def test_page_zero_is_kept(self):
        self.assertIn(("Page", "0"), format_source({**SOURCE, "page": 0}, False)["fields"])

    def test_private_paths_are_never_shown(self):
        self.assertEqual(display_filename(r"C:\Users\me\docs\guide.pdf"), "guide.pdf")
        self.assertEqual(display_filename("/home/me/guide.pdf"), "guide.pdf")
        self.assertIsNone(display_filename(None))

    def test_duplicate_sources_removed_in_rank_order(self):
        second = {**SOURCE, "number": 2, "chunk_id": "guide_p3_c3"}
        unique = dedupe_sources([SOURCE, second, dict(SOURCE)])
        self.assertEqual([s["number"] for s in unique], [1, 2])

    def test_format_sources_marks_cited_from_structured_citations(self):
        second = {**SOURCE, "number": 2, "chunk_id": "guide_p3_c3"}
        formatted = format_sources({
            "answer": "Mentions Guide.pdf but cites only [Source 2].",
            "sources": [SOURCE, second], "citations": [{"number": 2}]})
        self.assertEqual([(s["number"], s["cited"]) for s in formatted],
                         [(1, False), (2, True)])

    def test_excerpt_truncation(self):
        text = "word " * 500
        excerpt = truncate_excerpt(text)
        self.assertLessEqual(len(excerpt), MAX_EXCERPT_CHARS + 1)
        self.assertTrue(excerpt.endswith("…"))
        self.assertFalse(excerpt[:-1].endswith(" "))
        self.assertEqual(truncate_excerpt("  short\n text "), "short text")
        self.assertEqual(truncate_excerpt(None), "")
        self.assertEqual(len(truncate_excerpt("x" * 50, limit=10)), 11)

    def test_markdown_escaping(self):
        escaped = escape_markdown("[click](http://evil) **bold** <b>")
        self.assertNotIn("](", escaped.replace("\\]", ""))
        self.assertIn("\\[click\\]", escaped)
        self.assertIn("\\<b\\>", escaped)

    def test_answer_images_are_not_embedded(self):
        answer = "See ![secret](https://evil.example/?q=data) and **bold** [1]."
        self.assertEqual(safe_answer_markdown(answer),
                         "See secret and **bold** [1].")

    def test_metric_formatting_never_invents_zero(self):
        self.assertEqual(format_rate(None), "n/a")
        self.assertEqual(format_rate(0.6667), "66.7%")
        self.assertEqual(format_score(None), "n/a")
        self.assertEqual(format_score(0.63984), "0.640")


if __name__ == "__main__":
    unittest.main()
