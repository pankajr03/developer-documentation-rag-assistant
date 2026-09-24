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
```

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

Tests mock the OpenAI client and ChromaDB, so they are repeatable, run offline,
and consume no API credits.

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
├── tests/
│   ├── test_generator.py
│   └── test_ingestion.py
├── data/chroma/          persistent ChromaDB collection (git-ignored)
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
- Every retrieved chunk is listed as a source, including ones the answer did not
  cite. There is no relevance threshold yet.
