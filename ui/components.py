"""Streamlit building blocks shared by the pages."""

import logging
from typing import Any, Optional

import streamlit as st

from services.feedback_service import (
    MAX_COMMENT_LENGTH,
    RATING_THUMBS_DOWN,
    RATING_THUMBS_UP,
)
from ui import services, state
from ui.config import APP_TITLE, MAX_TOP_K, MIN_TOP_K
from ui.formatting import escape_markdown, safe_answer_markdown
from ui.services import KnowledgeBaseStatus

logger = logging.getLogger(__name__)

_BADGES = {
    "ready": ("Ready", "green", ":material/check_circle:"),
    "empty": ("Empty", "orange", ":material/inbox:"),
    "loading": ("Loading", "blue", ":material/hourglass_top:"),
    "error": ("Error", "red", ":material/error:"),
}


# --- Knowledge base ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _chroma_client(path: str) -> Any:
    """One ChromaDB client per database folder, shared across sessions.

    Only the client is cached. Answers and other per-user data are not.
    """
    import chromadb

    return chromadb.PersistentClient(path=path)


def refresh_status() -> KnowledgeBaseStatus:
    """Read the knowledge-base status and keep it for this script run."""
    from rag import vector_store

    try:
        # The path is fixed in rag.vector_store; the browser cannot choose it.
        path = vector_store.CHROMA_PATH
        client = _chroma_client(str(path)) if path.is_dir() else None
        status = services.knowledge_base_status(client)
    except Exception as error:
        logger.warning("Opening ChromaDB failed: %s", type(error).__name__)
        status = KnowledgeBaseStatus("error", message=services.MSG_VECTOR_DB)
    st.session_state["kb_status"] = status
    return status


def kb_status() -> KnowledgeBaseStatus:
    status = st.session_state.get("kb_status")
    return status if isinstance(status, KnowledgeBaseStatus) else refresh_status()


def status_badge(state_name: str) -> None:
    label, color, icon = _BADGES.get(state_name, _BADGES["error"])
    st.badge(label, color=color, icon=icon)


def kb_summary_line(status: KnowledgeBaseStatus) -> str:
    if status.state != "ready":
        return status.message
    text = f"{status.chunk_count:,} chunks"
    if status.document_count is not None:
        text += f" from {status.document_count:,} document(s)"
    return text


# --- Answers -------------------------------------------------------------------

def render_sources(message: dict[str, Any], show_excerpts: bool) -> None:
    sources = message.get("sources") or []
    if not sources:
        st.caption("No documentation chunks were retrieved.")
        return
    st.markdown("**Sources**")
    if message.get("is_fallback"):
        st.caption("These chunks were searched but did not support an answer.")
    for source in sources:  # already deduplicated, in retrieval order
        label = escape_markdown(source["label"])
        if source["cited"]:
            label += " · cited"
        with st.expander(label, icon=":material/description:"):
            for name, value in source["fields"]:
                st.markdown(f"**{name}:** {escape_markdown(value)}")
            if show_excerpts and source.get("excerpt"):
                # st.code renders document text literally: no Markdown or HTML.
                st.code(source["excerpt"], language=None, wrap_lines=True)


def _submit_feedback(message: dict[str, Any], rating: int) -> None:
    """Form callback: runs once per click, before the script reruns."""
    session = st.session_state
    response_id = message["response_id"]
    if state.feedback_rating(session, response_id) is not None:
        return  # already submitted in this session; never save twice
    comment = session.get(f"feedback_comment_{response_id}", "")
    try:
        services.save_answer_feedback(message, rating, comment, session["session_id"])
    except services.AssistantError as error:
        state.mark_feedback_failed(session, response_id, str(error))
        return
    state.mark_feedback_submitted(session, response_id, rating)


def render_feedback(message: dict[str, Any]) -> None:
    """Rating form for one answer. Keys derive from its stable response id."""
    session = st.session_state
    response_id = message["response_id"]
    rating = state.feedback_rating(session, response_id)
    if rating is not None:
        label = "Helpful" if rating == RATING_THUMBS_UP else "Not helpful"
        st.caption(f":material/check: Thanks, your feedback was saved ({label}).")
        return

    error = state.feedback_error(session, response_id)
    if error:
        st.error(error, icon=":material/error:")
    with st.form(key=f"feedback_form_{response_id}", border=False):
        st.caption("Was this answer helpful?")
        st.text_input(
            "Optional comment", key=f"feedback_comment_{response_id}",
            max_chars=MAX_COMMENT_LENGTH,
            placeholder="What was helpful, or what should be improved?")
        up, down = st.columns(2)
        up.form_submit_button(
            "👍 Helpful", key=f"feedback_up_{response_id}",
            on_click=_submit_feedback, args=(message, RATING_THUMBS_UP))
        down.form_submit_button(
            "👎 Not helpful", key=f"feedback_down_{response_id}",
            on_click=_submit_feedback, args=(message, RATING_THUMBS_DOWN))


def render_assistant_body(message: dict[str, Any]) -> None:
    if message.get("kind") == "error":
        st.error(message["content"], icon=":material/error:")
        return
    st.markdown(safe_answer_markdown(message["content"]))
    for warning in message.get("warnings") or []:
        st.warning(warning, icon=":material/warning:")
    render_sources(message, st.session_state.get("show_excerpts", True))
    if message.get("response_id"):
        render_feedback(message)


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["content"])  # the user's text, shown literally
        else:
            render_assistant_body(message)


# --- Sidebar -------------------------------------------------------------------

def _clear_conversation() -> None:
    state.clear_conversation(st.session_state)
    st.toast("Conversation cleared. Documents and feedback were kept.")


def render_sidebar(status: Optional[KnowledgeBaseStatus] = None) -> None:
    status = status or kb_status()
    info = services.model_info()
    with st.sidebar:
        st.markdown(f"### 📚 {APP_TITLE}")
        st.caption("Answers come only from the documents you index, "
                   "with numbered citations.")

        st.markdown("**Knowledge base**")
        status_badge(status.state)
        st.caption(kb_summary_line(status))
        st.caption(f"Collection `{info['collection']}` · "
                   f"embeddings `{info['embedding_model']}` · "
                   f"answers `{info['generation_model']}`")
        if not services.api_key_configured():
            st.warning(services.MSG_MISSING_KEY, icon=":material/key_off:")

        st.markdown("**Retrieval settings**")
        st.slider("Chunks to retrieve", MIN_TOP_K, MAX_TOP_K, key="top_k",
                  help="How many of the closest chunks the answer is based on.")
        st.toggle("Show source excerpts", key="show_excerpts")

        st.divider()
        st.button(
            "Clear conversation", key="clear_conversation",
            icon=":material/delete_sweep:",
            on_click=_clear_conversation, width="stretch",
            help="Removes this chat only. Documents, feedback and evaluation "
                 "reports are kept.")
