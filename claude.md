# Developer Documentation Assistant

## Project

This is a small learning project for building a RAG application incrementally. Keep each task narrow and review the implementation before adding the next feature. Do not build a complete production RAG system all at once.

## Current milestone

Stages 1-8 are implemented. The full RAG loop works: upload a document, index it
in ChromaDB, ask a question, and get an answer grounded in the retrieved chunks
with citations back to the source.

- Streamlit app in `app.py`; virtual environment at `.venv`; dependencies in `requirements.txt`
- `rag/loader.py` extracts text from TXT, Markdown, and PDF, one section per PDF page
- `rag/chunker.py` splits text into overlapping chunks, keeping page number and Markdown heading
- `rag/embeddings.py` creates embeddings with `text-embedding-3-small` and loads the API key from `.env`
- `rag/vector_store.py` stores chunks, embeddings, and metadata in the `developer_docs` ChromaDB collection at `data/chroma`
- `rag/retriever.py` embeds the question and returns the top K chunks with their metadata
- `rag/generator.py` builds numbered sources and answers with `gpt-5.4-mini` via the OpenAI Responses API
- Answers are grounded only in retrieved context; when the context does not support an answer the app returns a fixed fallback message
- Each stored chunk carries `source`, `document_id`, `chunk_id`, `chunk_index`, plus `page` and `section` when available; missing metadata is omitted, never invented
- The UI shows each source in an expander with filename, page, section, chunk id, distance, and excerpt
- Every answer has a UUID `response_id` (created in `generate_answer`). Users rate answers with thumbs up (1) or down (-1) plus an optional comment
- `services/feedback_service.py` stores feedback in SQLite at `data/feedback.db`, one row per `response_id` (upsert); `scripts/view_feedback.py` inspects it
- Tests live in `tests/` and mock the OpenAI client and ChromaDB, so they run offline

Chunks indexed before Stage 7 lack `page` and `chunk_id`. Use the Maintenance
expander in the app to reset the collection, then re-upload those documents.

See `README.md` for the architecture diagram, metadata reference, and an example
answer.

## Setup

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app runs at `http://localhost:8501` by default.

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py" -v
```

## Development guidance

- Prefer simple, readable implementations because this project is used to learn RAG concepts.
- Make one focused change per task.
- Preserve existing behavior unless the task explicitly changes it.
- Add or update focused tests when behavior becomes testable.
- Validate changes with the narrowest useful command before moving on.
- Avoid unrelated refactors and unnecessary dependencies.
- Keep secrets in `.env`; never commit API keys.

## Completed stages

1. Loader for TXT, Markdown, and PDF files returning extracted text and source metadata.
2. Document chunking.
3. Embeddings and a local vector store.
4. Retrieval for a user question.
5. Storing embeddings and metadata in ChromaDB, with retrieval reading from it.
6. Grounded answer generation from retrieved context only.
7. Source citations, page and chunk metadata, and source display in the interface.
8. Thumbs-up/down feedback on answers, saved in SQLite with a developer summary.

## Possible next tasks

Still one narrow change at a time.

- Filter out weakly matching chunks with a relevance threshold, instead of listing every retrieved chunk as a source.
- Show which sources the answer actually cited, separately from the ones merely searched.
- Detect section headings in PDFs, not just Markdown; today PDF chunks usually have no section name.
- Move from character-based chunking to sentence or paragraph boundaries so excerpts stop starting mid-sentence.
- Manage multiple indexed documents: list them, delete one, and re-index without resetting the whole collection.
- Evaluate answers using the saved feedback (Stage 9).
