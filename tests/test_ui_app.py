"""Streamlit AppTest smoke tests. ChromaDB and the feedback database are
redirected to temporary folders, and the RAG call is replaced by a fake."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from services.feedback_service import get_feedback_by_response_id
from ui.services import KnowledgeBaseStatus

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

ANSWER = {
    "content": "Report 30 minutes before race time [Source 1].",
    "question": "When do I report?",
    "response_id": "resp-test-1",
    "sources": [{"number": 1, "cited": True, "label": "[1] Schedule.pdf — page 1",
                 "fields": [("Document", "Schedule.pdf"), ("Page", "1")],
                 "excerpt": "Report 30 minutes before race time."}],
    "is_fallback": False,
    "warnings": [],
    "feedback_payload": {"sources": [], "retrieved_chunks": [],
                         "model_name": "fake-model", "retrieval_top_k": 3},
}


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.db = self.root / "feedback.db"
        for target, value in (
            ("rag.vector_store.CHROMA_PATH", self.root / "chroma"),
            ("services.feedback_service.FEEDBACK_DB_PATH", self.db),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        st.cache_resource.clear()

    def app(self):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        self.assertEqual(len(at.exception), 0, [e.message for e in at.exception])
        return at

    def test_starts_with_an_empty_knowledge_base(self):
        at = self.app()
        self.assertIn("No documentation is indexed yet", at.info[0].value)
        self.assertTrue(at.chat_input[0].disabled)
        self.assertFalse((self.root / "chroma").exists())  # status never creates it

    @patch("ui.services.knowledge_base_status",
           return_value=KnowledgeBaseStatus("ready", chunk_count=2))
    def test_ask_rerun_feedback_and_clear(self, _status):
        with patch("ui.services.ask_question", return_value=dict(ANSWER)) as ask:
            at = self.app()
            at.chat_input[0].set_value("  When do I report?  ").run()
            ask.assert_called_once_with("When do I report?", 3)
            self.assertEqual(len(at.chat_message), 2)
            self.assertTrue(any("30 minutes" in m.value for m in at.markdown))

            # Reruns redisplay the stored answer without calling the pipeline.
            at.run()
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(len(at.chat_message), 2)

            at.button(key="feedback_up_resp-test-1").click().run()
            self.assertEqual(ask.call_count, 1)
            row = get_feedback_by_response_id("resp-test-1", db_path=self.db)
            self.assertEqual(row["rating"], 1)
            self.assertEqual(at.session_state["feedback_status"],
                             {"resp-test-1": {"rating": 1}})
            # The form is replaced by a confirmation, so it cannot be sent twice.
            self.assertNotIn("feedback_up_resp-test-1",
                             [b.key for b in at.button])

            at.button(key="clear_conversation").click().run()
            self.assertEqual(at.session_state["messages"], [])
            self.assertIsNotNone(
                get_feedback_by_response_id("resp-test-1", db_path=self.db))

    @patch("ui.services.knowledge_base_status",
           return_value=KnowledgeBaseStatus("ready", chunk_count=2))
    def test_whitespace_question_is_not_sent(self, _status):
        with patch("ui.services.ask_question") as ask:
            at = self.app()
            at.chat_input[0].set_value("   ").run()
            ask.assert_not_called()
            self.assertEqual(at.session_state["messages"], [])


if __name__ == "__main__":
    unittest.main()
