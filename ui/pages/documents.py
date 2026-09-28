"""Manage Documents: upload, index and inspect the knowledge base."""

from dataclasses import asdict

import streamlit as st

from ui import components, services
from ui.config import ALLOWED_EXTENSIONS, MAX_FILES_PER_BATCH, MAX_UPLOAD_MB
from ui.validation import validate_upload

_STATUS_LABELS = {"indexed": "✅ Indexed", "skipped": "⏭️ Skipped", "failed": "❌ Failed"}


def _run_indexing(uploaded_files: list) -> None:
    """Index the selected files. Called only from the button click."""
    session = st.session_state
    status = components.kb_status()
    files = [(f.name, f.getvalue()) for f in uploaded_files]

    with st.status("Indexing documents…", expanded=True) as box:
        bar = st.progress(0.0)

        def progress(stage: str, index: int, count: int, name: str) -> None:
            box.update(label=f"{stage}: {name}" if name else stage)
            bar.progress(min(index / max(count, 1), 1.0))

        report = services.index_documents(
            files,
            known_document_ids=status.document_ids,
            known_content_hashes=status.content_hashes,
            progress=progress)
        failed = report.count("failed")
        box.update(label="Completed" if not failed else "Completed with errors",
                   state="error" if failed else "complete", expanded=False)

    session["last_index_report"] = [asdict(o) for o in report.outcomes]
    session["uploader_version"] += 1  # empty the uploader
    components.refresh_status()
    st.rerun()


def _render_last_report() -> None:
    outcomes = st.session_state.get("last_index_report")
    if not outcomes:
        return
    st.subheader("Last indexing result")
    indexed = [o for o in outcomes if o["status"] == "indexed"]
    cols = st.columns(4)
    cols[0].metric("Files indexed", len(indexed))
    cols[1].metric("Chunks created", sum(o["chunks"] for o in indexed))
    cols[2].metric("Skipped", sum(o["status"] == "skipped" for o in outcomes))
    cols[3].metric("Errors", sum(o["status"] == "failed" for o in outcomes))
    st.dataframe(
        [{"File": o["name"], "Result": _STATUS_LABELS[o["status"]],
          "Chunks": o["chunks"] or "", "Details": o["message"]} for o in outcomes],
        hide_index=True, width="stretch")


def _render_uploader() -> None:
    session = st.session_state
    st.subheader("Add documents")
    types = ", ".join(e.lstrip(".").upper() for e in ALLOWED_EXTENSIONS)
    uploaded = st.file_uploader(
        f"Upload {types} files (up to {MAX_UPLOAD_MB} MB each)",
        type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
        key=f"uploader_{session['uploader_version']}")

    too_many = len(uploaded) > MAX_FILES_PER_BATCH
    if uploaded:
        checks = [validate_upload(f.name, f.size) for f in uploaded]
        st.dataframe(
            [{"File": c.display_name, "Size": f"{f.size / 1024:,.0f} KB",
              "Check": "Ready" if c.ok else c.message}
             for f, c in zip(uploaded, checks)],
            hide_index=True, width="stretch")
        if too_many:
            st.warning(f"Select at most {MAX_FILES_PER_BATCH} files at a time.")
        st.caption("Indexing creates embeddings with the OpenAI API, which has "
                   "a small cost. Documents that are already indexed are skipped.")

    # Paid embedding calls happen only on this click, never on a rerun.
    if st.button("Index selected documents", type="primary",
                 icon=":material/upload_file:",
                 disabled=not uploaded or too_many):
        _run_indexing(uploaded)


def _render_indexed_documents(status: services.KnowledgeBaseStatus) -> None:
    st.subheader("Knowledge base")
    info = services.model_info()
    cols = st.columns(3)
    with cols[0]:
        st.caption("Status")
        components.status_badge(status.state)
    cols[1].metric("Chunks", f"{status.chunk_count:,}")
    cols[2].metric("Documents",
                   "n/a" if status.document_count is None else status.document_count)
    st.caption(f"Collection `{info['collection']}` · embedding model "
               f"`{info['embedding_model']}`")
    if status.state == "error":
        st.error(status.message, icon=":material/error:")
    elif status.documents:
        st.dataframe(
            [{"Document": d.name, "Document ID": d.document_id or "",
              "Chunks": d.chunks} for d in status.documents],
            hide_index=True, width="stretch")
    elif status.state == "empty":
        st.info("Nothing is indexed yet. Upload a document above.",
                icon=":material/inbox:")


def _render_maintenance() -> None:
    with st.expander("Maintenance", icon=":material/build:"):
        st.write("Resetting deletes **every indexed chunk** from ChromaDB. "
                 "Feedback and evaluation reports are not affected. Use it to "
                 "re-index a document, or chunks indexed before Stage 7.")
        confirmed = st.checkbox("I understand this deletes all indexed documents.")
        if st.button("Reset indexed documents", disabled=not confirmed,
                     icon=":material/delete_forever:"):
            try:
                services.reset_knowledge_base()
            except services.AssistantError as error:
                st.error(str(error))
            else:
                st.session_state["last_index_report"] = None
                components.refresh_status()
                st.rerun()


def render() -> None:
    st.title("Manage Documents")
    st.caption("Upload documentation and index it so the assistant can answer from it.")
    status = components.kb_status()
    _render_uploader()
    _render_last_report()
    _render_indexed_documents(status)
    _render_maintenance()
