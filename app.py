import streamlit as st

from rag.chunker import chunk_text
from rag.embeddings import create_embeddings
from rag.generator import NOT_FOUND_MESSAGE, generate_answer
from rag.loader import load_document
from rag.retriever import retrieve_chunks
from rag.vector_store import get_collection_count, store_chunks

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

st.write(
    "Upload project documentation and ask questions about your project."
)

uploaded_file = st.file_uploader(
    "Upload documentation",
    type=["txt", "md", "pdf"]
)

if uploaded_file:
    try:
        extracted_text = load_document(uploaded_file)
        if not extracted_text.strip():
            raise ValueError("The document contains no extractable text.")

        chunks = chunk_text(extracted_text)
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

        for index, chunk in enumerate(chunks, start=1):
            with st.expander(f"Chunk {index}"):
                st.text(chunk)

        if st.button("Create Embeddings"):
            try:
                st.session_state.embeddings = create_embeddings(chunks)
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
        st.session_state.qa_result = {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "answer": None,
        }
        try:
            st.session_state.qa_result["answer"] = generate_answer(
                question,
                retrieved_chunks,
            )
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        except Exception:
            st.error("Could not generate an answer. Please try again.")

qa_result = st.session_state.qa_result
if qa_result is not None:
    st.write("Question:", qa_result["question"])

    if qa_result["answer"] is not None:
        st.subheader("Answer")
        st.write(qa_result["answer"])

    # Sources come straight from ChromaDB metadata, never from the LLM output.
    st.subheader("Sources")
    if qa_result["answer"] == NOT_FOUND_MESSAGE:
        st.caption("The answer was not found in these retrieved chunks.")
    if not qa_result["retrieved_chunks"]:
        st.write("No documentation chunks were retrieved.")

    seen_sources = set()
    for result in qa_result["retrieved_chunks"]:
        source_key = (result["source"], result["chunk_index"])
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        with st.expander(f"{result['source']} — Chunk {result['chunk_index']}"):
            st.write(f"Distance: {result['distance']}")
            st.text(result["text"])
