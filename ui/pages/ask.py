"""Ask Documentation: the chat page."""

import streamlit as st

from ui import components, services, state
from ui.config import APP_TAGLINE, APP_TITLE, MAX_QUESTION_CHARS
from ui.validation import QuestionValidationError, validate_question


def _answer(question: str) -> None:
    """Run the real pipeline once for a newly submitted question.

    The result is stored in the chat history, so later reruns (opening a
    source, rating the answer) only redisplay it and never call the LLM again.
    """
    session = st.session_state
    components.render_message(state.add_user_message(session, question))
    with st.chat_message("assistant"):
        with st.spinner("Searching the documentation and writing an answer…"):
            try:
                answer = services.ask_question(question, int(session["top_k"]))
            except services.AssistantError as error:
                message = state.add_error_message(session, question, str(error))
            else:
                message = state.add_assistant_message(session, answer)
        components.render_assistant_body(message)


def render() -> None:
    session = st.session_state
    status = components.kb_status()

    title, badge = st.columns([5, 1], vertical_alignment="center")
    title.title(APP_TITLE)
    with badge:
        components.status_badge(status.state)
    st.caption(APP_TAGLINE)

    if status.state == "empty":
        st.info("No documentation is indexed yet. Open **Manage Documents** "
                "in the sidebar to add some, then come back to ask questions.",
                icon=":material/inbox:")
    elif status.state == "error":
        st.error(status.message, icon=":material/error:")

    for message in session["messages"]:
        components.render_message(message)

    if not session["messages"] and status.state == "ready":
        st.caption("Try a question such as “What is the reporting time?” "
                   "Each answer lists the chunks it was based on.")

    prompt = st.chat_input(
        "Ask a question about your documentation",
        max_chars=MAX_QUESTION_CHARS,
        disabled=status.state != "ready")
    if prompt is None:
        return
    try:
        question = validate_question(prompt)
    except QuestionValidationError as error:
        st.warning(str(error), icon=":material/edit:")
        return
    _answer(question)
