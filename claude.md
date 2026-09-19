# Developer Documentation Assistant

## Project

This is a small learning project for building a RAG application incrementally. Keep each task narrow and review the implementation before adding the next feature. Do not build a complete production RAG system all at once.

## Current milestone

- Streamlit app created in `app.py`
- Python virtual environment created at `.venv`
- Dependencies recorded in `requirements.txt`
- Streamlit UI accepts TXT, Markdown, and PDF uploads
- The current UI echoes the submitted question; document processing and retrieval are not implemented yet

## Setup

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app runs at `http://localhost:8501` by default.

## Development guidance

- Prefer simple, readable implementations because this project is used to learn RAG concepts.
- Make one focused change per task.
- Preserve existing behavior unless the task explicitly changes it.
- Add or update focused tests when behavior becomes testable.
- Validate changes with the narrowest useful command before moving on.
- Avoid unrelated refactors and unnecessary dependencies.
- Keep secrets in `.env`; never commit API keys.

## Planned incremental tasks

1. Implement a loader for TXT, Markdown, and PDF files that returns extracted text and source metadata.
2. Add document chunking.
3. Add embeddings and a local vector store.
4. Implement retrieval for a user question.
5. Add an LLM response step with source citations.
6. Improve the Streamlit interface and error handling.
