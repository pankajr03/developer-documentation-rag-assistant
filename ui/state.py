"""Per-browser-session state, kept in st.session_state.

The helpers take any MutableMapping so tests can use a plain dict. Messages
hold only plain data (strings, numbers, lists, dicts), never client objects.

Assistant message shape:
    {"id", "role": "assistant", "kind": "answer" | "error", "content",
     "question", "response_id", "sources", "is_fallback", "warnings",
     "feedback": {...}}

An answer's response_id comes from rag.generator.generate_answer, which runs
once when the question is submitted. Reruns only re-display stored messages,
so they never call the LLM again or change the id.
"""

from typing import Any, MutableMapping, Optional
from uuid import uuid4

from ui.config import DEFAULT_TOP_K

State = MutableMapping[str, Any]


def init_state(state: State) -> None:
    """Add missing keys without touching existing ones."""
    defaults = {
        "messages": [],
        "feedback_status": {},       # response_id -> {"rating"} or {"error"}
        "session_id": str(uuid4()),  # identifies this browser session only
        "top_k": DEFAULT_TOP_K,
        "show_excerpts": True,
        "last_index_report": None,
        # Changing the uploader's key empties it after indexing, so the same
        # files cannot be indexed twice by accident.
        "uploader_version": 0,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def add_user_message(state: State, question: str) -> dict[str, Any]:
    message = {"id": str(uuid4()), "role": "user", "content": question}
    state["messages"].append(message)
    return message


def add_assistant_message(state: State, answer: dict[str, Any]) -> dict[str, Any]:
    """Store an answer built by ui.services.ask_question."""
    message = {"id": str(uuid4()), "role": "assistant", "kind": "answer", **answer}
    state["messages"].append(message)
    return message


def add_error_message(state: State, question: str, text: str) -> dict[str, Any]:
    """Store a user-safe error in place of an answer. It gets no feedback."""
    message = {"id": str(uuid4()), "role": "assistant", "kind": "error",
               "content": text, "question": question, "response_id": None}
    state["messages"].append(message)
    return message


def feedback_rating(state: State, response_id: str) -> Optional[int]:
    """The rating already submitted for this answer in this session, if any."""
    return state["feedback_status"].get(response_id, {}).get("rating")


def feedback_error(state: State, response_id: str) -> Optional[str]:
    return state["feedback_status"].get(response_id, {}).get("error")


def mark_feedback_submitted(state: State, response_id: str, rating: int) -> bool:
    """Record a submitted rating. Returns False if one was already recorded."""
    if feedback_rating(state, response_id) is not None:
        return False
    state["feedback_status"][response_id] = {"rating": rating}
    return True


def mark_feedback_failed(state: State, response_id: str, text: str) -> None:
    state["feedback_status"][response_id] = {"error": text}


def clear_conversation(state: State) -> None:
    """Forget this session's chat only.

    Documents, ChromaDB, saved feedback and evaluation reports are untouched,
    and retrieval settings are kept.
    """
    state["messages"] = []
    state["feedback_status"] = {}
