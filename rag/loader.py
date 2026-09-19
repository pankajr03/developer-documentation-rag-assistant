from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def load_document(uploaded_file):
    """Extract text from a Streamlit uploaded TXT, Markdown, or PDF file."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if file_extension in {".txt", ".md"}:
        return file_bytes.decode("utf-8")

    if file_extension == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(
        f"Unsupported file type: {file_extension or 'unknown'}. "
        "Please upload a PDF, TXT, or Markdown file."
    )
