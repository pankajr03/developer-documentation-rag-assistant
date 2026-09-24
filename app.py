import logging
from uuid import uuid4

import streamlit as st

from rag.chunker import chunk_sections
from rag.embeddings import create_embeddings
from rag.generator import FALLBACK_MESSAGE, generate_answer
from rag.loader import load_document, section_text
from rag.retriever import retrieve_chunks
from rag.vector_store import get_collection_count, reset_collection, store_chunks
from services.feedback_service import (
    MAX_COMMENT_LENGTH,
    RATING_THUMBS_DOWN,
    RATING_THUMBS_UP,
    FeedbackStorageError,
    get_feedback_by_response_id,
    get_feedback_summary,
    save_feedback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(
    page_title="Developer Documentation Assistant",
    page_icon="📚"
)

st.title("📚 Developer Documentation Assistant")

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "document_key" not in st.session_state:
    st.session_state.document_key = None
if "stored_document" not in st.session_state:
    st.session_state.stored_document = None
if "qa_result" not in st.session_state:
    st.session_state.qa_result = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())


def render_feedback(result: dict) -> None:
    """Thumbs up/down plus an optional comment for one generated answer.

    Widget keys include the response id, so every answer gets its own widgets
    and reruns never mix ratings between answers. Submitting only writes to
    SQLite; it never calls the retriever or the LLM.
    """
    response_id = result["response_id"]
    rating_key = f"feedback_rating_{response_id}"
    comment_key = f"feedback_comment_{response_id}"

    try:
        saved = get_feedback_by_response_id(response_id)
    except FeedbackStorageError as error:
        saved = None
        st.warning(str(error))

    # Show what is already saved for this answer. Only done before the widgets
    # exist, so it never overwrites a choice the user is making.
    if saved and rating_key not in st.session_state:
        st.session_state[rating_key] = (
            1 if saved["rating"] == RATING_THUMBS_UP else 0)
    if saved and comment_key not in st.session_state:
        st.session_state[comment_key] = saved["comment"] or ""

    st.write("Was this answer helpful?")
    # st.feedback("thumbs") returns 1 for thumbs up, 0 for thumbs down, or None.
    selection = st.feedback("thumbs", key=rating_key)
    comment = st.text_area(
        "Optional feedback",
        key=comment_key,
        max_chars=MAX_COMMENT_LENGTH,
        placeholder="Write what was helpful or what should be improved",
    )

    if st.button(
        "Submit feedback",
        key=f"feedback_submit_{response_id}",
        disabled=selection is None,
    ):
        try:
            save_feedback(
                response_id=response_id,
                question=result["question"],
                answer=result["answer"],
                rating=RATING_THUMBS_UP if selection == 1 else RATING_THUMBS_DOWN,
                comment=comment,
                sources=result["sources"],
                retrieved_chunks=result["retrieved_chunks"],
                model_name=result["model_name"],
                retrieval_top_k=result["retrieval_top_k"],
                session_id=st.session_state.session_id,
            )
            st.success(
                "Feedback updated." if saved else "Thank you, feedback saved.")
        except ValueError as error:
            st.error(str(error))
        except FeedbackStorageError as error:
            st.error(str(error))
    elif saved:
        st.caption("Feedback saved. Change it and submit again to update it.")


st.write(
    "Upload project documentation and ask questions about your project."
)

uploaded_file = st.file_uploader(
    "Upload documentation",
    type=["txt", "md", "pdf"]
)

if uploaded_file:
    try:
        sections = load_document(uploaded_file)
        extracted_text = section_text(sections)
        if not extracted_text.strip():
            raise ValueError("The document contains no extractable text.")

        chunks = chunk_sections(sections)
        if not chunks:
            raise ValueError("The document did not produce any text chunks.")

        document_key = (uploaded_file.name, uploaded_file.size)
        if st.session_state.document_key != document_key:
            st.session_state.document_key = document_key
            st.session_state.embeddings = None
            st.session_state.stored_document = None

        st.success("Document loaded successfully.")
        st.write(f"Total extracted characters: {len(extracted_text)}")
        st.write(f"Total chunks: {len(chunks)}")

        for chunk in chunks:
            label = f"Chunk {chunk['chunk_index']}"
            if chunk["page"] is not None:
                label += f" — Page {chunk['page']}"
            if chunk["section"]:
                label += f" — {chunk['section']}"
            with st.expander(label):
                st.text(chunk["text"])

        if st.button("Create Embeddings"):
            try:
                st.session_state.embeddings = create_embeddings(
                    [chunk["text"] for chunk in chunks]
                )
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Embedding generation failed: {error}")

        embeddings = st.session_state.embeddings
        if embeddings is not None:
            st.write(f"Total embeddings: {len(embeddings)}")
            st.write(f"Embedding dimensions: {len(embeddings[0])}")
            st.write("First embedding preview:")
            st.code(str(embeddings[0][:10]))

        if st.button("Store in Vector Database"):
            if embeddings is None:
                st.error("Create embeddings before storing the document.")
            else:
                try:
                    stored_count = store_chunks(
                        chunks,
                        embeddings,
                        uploaded_file.name,
                    )
                    st.session_state.stored_document = {
                        "source": uploaded_file.name,
                        "stored_count": stored_count,
                    }
                except Exception as error:
                    st.error(
                        f"Could not store the document in ChromaDB: {error}")

        stored_document = st.session_state.stored_document
        if stored_document is not None:
            st.success(
                f"{stored_document['stored_count']} chunks stored in ChromaDB."
            )
            st.write("Collection: developer_docs")
            st.write(f"Stored chunks: {get_collection_count()}")
            st.write(f"Source: {stored_document['source']}")
    except Exception as error:
        st.error(f"Could not process the document: {error}")

st.subheader("Ask Documentation")
question = st.text_input("Ask a question about the documentation")
top_k = st.number_input(
    "Number of retrieved chunks",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
)

# The LLM is only called when the button is clicked. The result is kept in
# session_state so reruns (e.g. opening an expander) do not repeat API calls.
if st.button("Ask"):
    st.session_state.qa_result = None
    try:
        retrieved_chunks = retrieve_chunks(question, top_k=top_k)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
    except Exception:
        st.error(
            "Could not search the documentation. Check your API key and try again.")
    else:
        try:
            st.session_state.qa_result = generate_answer(
                question,
                retrieved_chunks,
                top_k=top_k,
            )
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        except Exception:
            st.error("Could not generate an answer. Please try again.")

qa_result = st.session_state.qa_result
if qa_result is not None:
    st.write("Question:", qa_result["question"])

    st.subheader("Answer")
    st.write(qa_result["answer"])

    render_feedback(qa_result)

    # Sources come straight from ChromaDB metadata, never from the LLM output.
    st.subheader("Sources")
    if qa_result["answer"] == FALLBACK_MESSAGE:
        st.caption(
            "These chunks were searched but do not support an answer."
        )
    if not qa_result["sources"]:
        st.write("No documentation chunks were retrieved.")

    for source in qa_result["sources"]:
        page_label = (
            f"Page {source['page']}" if source["page"] is not None
            else "Page not available"
        )
        with st.expander(
            f"Source {source['number']} — {source['filename']} — {page_label}"
        ):
            st.write(f"File: {source['filename']}")
            st.write(f"Page: {source['page'] if source['page'] is not None else 'Not available'}")
            if source["section"]:
                st.write(f"Section: {source['section']}")
            st.write(f"Chunk ID: {source['chunk_id'] or 'Not available'}")
            if source["distance"] is not None:
                st.write(f"Distance: {source['distance']}")
            st.text(source["content"])

with st.sidebar.expander("Developer: feedback summary"):
    try:
        summary = get_feedback_summary()
        st.metric("Total ratings", summary["total"])
        st.metric("Thumbs up", summary["thumbs_up"])
        st.metric("Thumbs down", summary["thumbs_down"])
        st.metric("Positive ratings", f"{summary['positive_percentage']}%")
    except FeedbackStorageError as error:
        st.warning(str(error))

with st.expander("Maintenance"):
    st.write(
        "Chunks indexed before Stage 7 have no page number or chunk id. "
        "Reset the collection and re-upload those documents to add the "
        "citation metadata."
    )
    st.write(f"Stored chunks: {get_collection_count()}")
    if st.button("Reset Indexed Documents"):
        try:
            reset_collection()
            st.session_state.stored_document = None
            st.session_state.qa_result = None
            st.success("The collection was reset. Re-upload your documents.")
        except Exception as error:
            st.error(f"Could not reset the collection: {error}")
