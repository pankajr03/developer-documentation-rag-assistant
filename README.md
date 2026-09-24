# Developer Documentation Assistant

A small RAG (Retrieval-Augmented Generation) application built incrementally as
a learning project. Upload TXT, Markdown, or PDF documentation, index it in
ChromaDB, and ask questions that are answered **only** from the indexed text,
with citations back to the exact chunk that supports each statement.

## Pipeline

```text
INDEXING                          QUERY

Document                          User question
   |                                 |
load_document()   -> page metadata  create_embedding()
   |                                 |
chunk_sections()  -> page, section  ChromaDB similarity search
   |                                 |
create_embeddings()                 Top K chunks + metadata
   |                                 |
ChromaDB (text + metadata)          generate_answer()
                                     |
                                    Grounded answer + numbered sources
                                     |
                                    Thumbs up/down + comment -> SQLite
```

## What Stage 8 adds

Stage 8 lets a user rate each generated answer so the system can be evaluated
and improved later:

- **Thumbs up / thumbs down** directly under every answer, with an optional
  comment and a **Submit feedback** button (disabled until a rating is chosen).
- **Persistent storage** in SQLite (`data/feedback.db`), not in session state,
  so feedback survives restarts.
- **A stable `response_id`** (UUID) for every answer.
- **One row per answer.** Changing a rating or comment updates the existing row.
- **Enough context to evaluate later:** question, answer, sources, retrieved
  chunk metadata, model name, and `top_k`.
- **A developer summary** in the sidebar and a command-line viewer.

Feedback code lives in `services/feedback_service.py`; `app.py` only draws the
widgets and calls that module.

### Response IDs

`generate_answer()` creates `response_id = str(uuid4())` once, when the answer
is generated, and returns it inside the result. The result is kept in
`st.session_state.qa_result`, so Streamlit reruns (opening an expander, typing a
comment, clicking a thumb) reuse the same result and the same id. A new id is
made only when **Ask** produces a new answer. All feedback widget keys include
the id (`feedback_rating_<id>`, `feedback_comment_<id>`, `feedback_submit_<id>`),
so two answers can never share widget state.

The result now looks like:

```python
{
    "response_id": "6f0c1b52-...",
    "question": "...",
    "answer": "...",
    "sources": [...],            # numbered sources, as in Stage 7
    "retrieved_chunks": [...],   # what the retriever returned
    "model_name": "gpt-5.4-mini",
    "retrieval_top_k": 3,
}
```

### How to submit and update feedback

1. Ask a question and read the answer.
2. Click the thumbs-up (helpful) or thumbs-down (not helpful) under **Was this
   answer helpful?**
3. Optionally write a comment, then click **Submit feedback**.
4. To change your mind, pick the other thumb and/or edit the comment and click
   **Submit feedback** again. The saved row is updated, not duplicated.

Submitting feedback only writes to SQLite. It does not call the retriever or the
LLM, and the answer and sources stay on screen.

### Feedback database

Location: `data/feedback.db` (created automatically, git-ignored together with
its `-shm`, `-wal` and `-journal` files).

```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id TEXT NOT NULL UNIQUE,
    session_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (1, -1)),
    comment TEXT,
    sources_json TEXT,
    retrieved_chunks_json TEXT,
    model_name TEXT,
    retrieval_top_k INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT
)
```

| Rating | Meaning |
| --- | --- |
| `1` | thumbs up: the answer was helpful |
| `-1` | thumbs down: the answer was not helpful |

- `response_id` is `UNIQUE`, and saving uses `INSERT ... ON CONFLICT(response_id)
  DO UPDATE`, so a response can only ever have one current rating. On update
  only `rating`, `comment` and `updated_at` change; `created_at`, the question,
  the answer and the stored sources keep their original values.
- `updated_at` is `NULL` until the first change. Timestamps are UTC ISO-8601.
- `session_id` identifies the Streamlit browser session that gave the rating.
- `sources_json` holds each source's number, filename, page, chunk id,
  document id, section, distance, and an `excerpt` cut to 500 characters.
- `retrieved_chunks_json` holds retrieved-chunk **metadata only** (no text).
- All SQL uses `?` parameters. No user input is concatenated into a query.

### Inspect saved feedback

Sidebar: open **Developer: feedback summary** for total ratings, thumbs-up and
thumbs-down counts, and the positive-rating percentage.

Command line (newest first; `--limit` accepts 1-100, default 10):

```powershell
.\.venv\Scripts\python.exe scripts\view_feedback.py
.\.venv\Scripts\python.exe scripts\view_feedback.py --limit 20
```

```text
Total ratings: 2  |  Thumbs up: 1  |  Thumbs down: 1  |  Positive: 50.0%

Most recent 2 record(s):

ID:           2
Response ID:  b8d4a7c0-2f61-4c1e-9a3b-5e7d0c1f8a44
Question:     Does the schedule mention overtime pay?
Rating:       thumbs down (-1)
Comment:      The requested information was unavailable in the document.
Created at:   2026-03-14T09:31:07.482913+00:00
Updated at:   -
Sources:      sample_speed_schedule.pdf
...
```

The viewer never prints answers, excerpts, chunk text, or anything from `.env`.

### Example feedback record (fictional)

```json
{
  "id": 1,
  "response_id": "3f1c2a9e-7d54-4b0a-8e21-9c6d4a5b7e10",
  "session_id": "d2b8e6f4-0a13-4c77-b5a9-1e8f3c2d6a90",
  "question": "What is the reporting-time rule mentioned in the Speed Schedule?",
  "answer": "Every crew member must report at least 45 minutes before departure. [Source 1]",
  "rating": 1,
  "comment": "The answer was clear and the correct schedule source was shown.",
  "sources_json": "[{\"number\": 1, \"filename\": \"sample_speed_schedule.pdf\", \"page\": 3, \"chunk_id\": \"sample_speed_schedule_p3_c2\", \"document_id\": \"sample_speed_schedule\", \"distance\": 0.844, \"excerpt\": \"Section 3: Reporting Time  Every crew member must report...\"}]",
  "retrieved_chunks_json": "[{\"source\": \"sample_speed_schedule.pdf\", \"page\": 3, \"chunk_id\": \"sample_speed_schedule_p3_c2\", \"chunk_index\": 2, \"document_id\": \"sample_speed_schedule\", \"distance\": 0.844}]",
  "model_name": "gpt-5.4-mini",
  "retrieval_top_k": 3,
  "created_at": "2026-03-14T09:12:44.103552+00:00",
  "updated_at": null
}
```

### Privacy considerations

- Feedback is stored locally in `data/feedback.db`. It is not sent anywhere and
  is not shown in the app; the UI only shows aggregate counts.
- The question and answer are stored, so they may contain private document
  content. Treat `data/feedback.db` like the documents themselves.
- Stored excerpts are cut to 500 characters, comments are limited to 1000
  characters (longer ones are rejected), questions are cut to 2000 and answers
  to 8000. The limits are constants at the top of
  `services/feedback_service.py`.
- Never stored: API keys, environment variables, embedding vectors, tokens,
  whole documents, or model reasoning. Source and chunk records are copied field
  by field from an allow-list, so extra keys are dropped.
- Comments are Unicode-normalized (NFC) and control characters are removed.
- Delete the database file to erase all feedback.

## What Stage 7 adds

Stage 6 produced a grounded answer as a plain string. Stage 7 makes every
answer verifiable:

- **Source metadata end to end.** PDF page numbers survive loading and chunking,
  and each stored chunk gets a unique `chunk_id` plus a `document_id`.
- **Numbered sources.** Retrieved chunks become `[Source 1]`, `[Source 2]`, …
  The numbering follows retrieval order and stays stable for one request.
- **Inline citations.** The model is instructed to cite facts as `[Source 1]`.
  Citations naming a source number that was not retrieved are stripped out and
  logged, so the answer can never cite a document that was not searched.
- **Structured results.** `generate_answer()` returns
  `{"question", "answer", "sources"}` instead of a string.
- **Source display.** Each source appears in its own Streamlit expander with
  filename, page, section, chunk id, distance, and the supporting excerpt.
- **Deduplication.** Results sharing the same filename, page, *and* chunk id are
  shown once; different chunks from the same page are both kept.

## How source metadata is stored

Each ChromaDB record carries:

| Field | Source | Notes |
| --- | --- | --- |
| `source` | uploaded filename | always present |
| `document_id` | slug of the filename | e.g. `speed_schedule` |
| `chunk_id` | `{document_id}_p{page}_c{chunk_index}` | the record's Chroma id |
| `chunk_index` | position in the document | counts across the whole file |
| `page` | PDF page number (1-based) | **omitted** for TXT/Markdown |
| `section` | nearest Markdown heading | **omitted** when there is none |

Missing metadata is never invented. Optional fields are left out of the record
and shown as `Not available` in the interface.

## Re-indexing existing documents

**Yes, documents indexed before Stage 7 must be re-indexed.** Their records have
only `source` and `chunk_index`, so they display as `Page: Not available` and
`Chunk ID: Not available`. They still retrieve and still produce answers.

To migrate, open the **Maintenance** expander at the bottom of the app, click
**Reset Indexed Documents**, then re-upload each document and run *Create
Embeddings* → *Store in Vector Database*. Re-indexing costs embedding API calls
for the re-uploaded text.

## How citations are displayed

The answer contains inline markers such as `[Source 1]`. Below it, the
**Sources** section lists the matching numbered sources:

```text
Source 1 — speed_schedule.pdf — Page 3
    File: speed_schedule.pdf
    Page: 3
    Chunk ID: speed_schedule_p3_c2
    Distance: 0.844
    Section 3: Reporting Time ...
```

Source details come from ChromaDB metadata, never from the model, so a filename
or page number cannot be hallucinated. When the answer is the fallback message,
the searched sources are still listed, labeled as not supporting an answer.

## Setup

PowerShell, from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put your key in `.env` (never commit it):

```text
OPENAI_API_KEY=sk-...
```

## Run the application

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Or without activating the virtual environment:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The app runs at http://localhost:8501.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py" -v
```

Or a single module:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_generator -v
```

Feedback tests only:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_feedback_service -v
```

Tests mock the OpenAI client and ChromaDB, so they are repeatable, run offline,
and consume no API credits. Feedback tests use a temporary SQLite file and never
touch `data/feedback.db`.

## Example response

Question:

```text
What is the reporting-time rule mentioned in the Speed Schedule?
```

Answer (produced from a 4-page sample schedule PDF; the text comes entirely from
the indexed document):

```text
Every crew member must report to the depot supervisor at least 45 minutes
before the scheduled departure time. If they report later, they are marked
late and the duty officer reassigns the trip. [Source 1]
```

Sources:

```text
Source 1 — sample_speed_schedule.pdf — Page 3
    Chunk ID: sample_speed_schedule_p3_c2   Distance: 0.844
    "Section 3: Reporting Time  Every crew member must report to the depot
     supervisor at least 45 minutes before the scheduled departure time..."

Source 2 — sample_speed_schedule.pdf — Page 1
    Chunk ID: sample_speed_schedule_p1_c0   Distance: 0.930

Source 3 — sample_speed_schedule.pdf — Page 4
    Chunk ID: sample_speed_schedule_p4_c3   Distance: 1.196
```

An unsupported question against the same document returns:

```text
I could not find this information in the uploaded documents.
```

## Project structure

```text
developer-doc-assistant/
├── rag/
│   ├── loader.py         extract text + page metadata
│   ├── chunker.py        chunk text, keep page and heading
│   ├── embeddings.py     OpenAI embeddings + API key handling
│   ├── vector_store.py   ChromaDB storage, chunk ids, reset
│   ├── retriever.py      similarity search, returns metadata
│   └── generator.py      numbered sources, grounded answer, citations
├── services/
│   └── feedback_service.py  SQLite feedback store (upsert, summary, listing)
├── scripts/
│   └── view_feedback.py     developer viewer for saved feedback
├── tests/
│   ├── test_feedback_service.py
│   ├── test_generator.py
│   └── test_ingestion.py
├── data/chroma/          persistent ChromaDB collection (git-ignored)
├── data/feedback.db      feedback ratings (git-ignored)
├── app.py                Streamlit interface
├── .env                  OPENAI_API_KEY (git-ignored)
└── requirements.txt
```

## Models

| Purpose | Model | Set in |
| --- | --- | --- |
| Embeddings | `text-embedding-3-small` | `rag/embeddings.py` (`MODEL_NAME`) |
| Answer generation | `gpt-5.4-mini` | `rag/generator.py` (`MODEL_NAME`) |

## Known limitations

- `section` is detected from Markdown headings only, so PDF chunks usually have
  no section name.
- Chunking is character-based and splits at fixed offsets, so a chunk can begin
  mid-sentence.
- Feedback is stored per answer, not per source, so a rating cannot say which
  cited source was wrong. There is no login, so `session_id` only identifies a
  browser session, and anyone who can open the app can rate.
- Every retrieved chunk is listed as a source, including ones the answer did not
  cite. There is no relevance threshold yet.
