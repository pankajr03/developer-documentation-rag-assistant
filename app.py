import streamlit as st

from rag.chunker import chunk_text
from rag.loader import load_document

st.set_page_config(
    page_title="Developer Documentation Assistant",
    page_icon="📚"
)

st.title("📚 Developer Documentation Assistant")

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
        chunks = chunk_text(extracted_text)

        st.success("Document loaded successfully.")
        st.write(f"Total extracted characters: {len(extracted_text)}")
        st.write(f"Total chunks: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            with st.expander(f"Chunk {index}"):
                st.text(chunk)
    except Exception as error:
        st.error(f"Could not process the document: {error}")

question = st.text_input(
    "Ask a question about the documentation"
)

if question:
    st.write("Question:", question)
