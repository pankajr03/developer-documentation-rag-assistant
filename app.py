import streamlit as st

from rag.chunker import chunk_text
from rag.embeddings import create_embeddings
from rag.loader import load_document
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

question = st.text_input(
    "Ask a question about the documentation"
)

if question:
    st.write("Question:", question)
