"""Streamlit entry point: page setup, sidebar and navigation.

Start with:  streamlit run app.py
The pages live in ui/pages/ and call ui/services.py, which wraps the real
rag/, services/ and evaluation/ modules.
"""

import logging

import streamlit as st

from ui import components, state
from ui.config import APP_TITLE
from ui.pages import ask, documents, evaluation, feedback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(page_title=APP_TITLE, page_icon="📚")

state.init_state(st.session_state)
status = components.refresh_status()

navigation = st.navigation([
    st.Page(ask.render, title="Ask Documentation", icon=":material/chat:",
            url_path="ask", default=True),
    st.Page(documents.render, title="Manage Documents",
            icon=":material/folder_open:", url_path="documents"),
    st.Page(feedback.render, title="Feedback", icon=":material/thumbs_up_down:",
            url_path="feedback"),
    st.Page(evaluation.render, title="Evaluation", icon=":material/analytics:",
            url_path="evaluation"),
])
components.render_sidebar(status)
navigation.run()
