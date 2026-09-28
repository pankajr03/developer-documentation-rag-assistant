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

## What Stage 11 adds

A polished **Streamlit** interface. It replaces the single-page app with four
pages and a sidebar, and reuses the same RAG, feedback and evaluation code.
Streamlit was kept because the app already used it, and version 1.64 has
built-in navigation, chat, status and badge components, so no new dependency
was needed.

```text
Sidebar                        Pages
├── About + models             ├── Ask Documentation  chat with cited answers
├── Knowledge-base status      ├── Manage Documents   upload, index, list, reset
├── Retrieval settings         ├── Feedback           ratings summary (read-only)
└── Clear conversation         └── Evaluation         Stage 10 reports (read-only)
```

UI code lives in `ui/` and never talks to OpenAI, ChromaDB or SQLite directly:

- `ui/services.py`: the boundary. It calls `retrieve_chunks`, `generate_answer`,
  `create_embeddings`, `store_chunks`, `save_feedback` and turns every error
  into a short, user-safe message.
- `ui/state.py`: chat history and feedback state in `st.session_state`.
- `ui/validation.py`: question and upload validation, filename sanitising.
- `ui/formatting.py`: source display data, excerpt truncation, safe Markdown.
- `ui/reports.py`: read-only, path-safe access to evaluation reports.
- `ui/components.py`: shared widgets (sources, feedback form, sidebar).
- `ui/pages/`: one module per page. `app.py` only sets up navigation.
- `ui/config.py`: limits and defaults (question length, upload size, `top_k`).

### Start the app

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open http://localhost:8501. The only required environment variable is
`OPENAI_API_KEY` in `.env` (see [Setup](#setup)). If it is missing, the app
still starts and shows a warning; indexing and asking are the only actions
that need it.

### Supported documents and indexing

PDF, TXT (UTF-8) and Markdown, up to **10 MB** each and 10 files per batch.
On **Manage Documents**:

1. Choose one or more files. Each is checked immediately (type, size, empty).
2. Click **Index selected documents**. Nothing is embedded until you click,
   so Streamlit reruns never cause paid API calls.
3. A status box shows *Reading document → Creating chunks → Generating
   embeddings → Saving to ChromaDB → Completed*.
4. The result table lists each file as indexed, skipped or failed, with the
   reason, plus files indexed, chunks created, skipped and errors. The
   uploader is emptied so the same files are not indexed twice by accident.

Duplicates are **skipped, never replaced**: a file whose document id (derived
from its name, as before) is already indexed, or whose SHA-256 content hash
matches an indexed file (new documents store a `content_hash` in their chunk
metadata). To re-index a document, use **Maintenance → Reset indexed
documents** (you must tick a confirmation box first; this deletes every
indexed chunk but not feedback or reports), then upload again. Scanned PDFs
without a text layer are skipped with "No usable text found".

### Asking questions and citations

On **Ask Documentation**, type into the chat box. Each question is checked
(not empty, at most 1,000 characters), then the real retriever and generator
run once. The answer, its sources and its feedback form are stored in the
session, so opening a source or rating an answer never calls the model again.

Under each answer, **Sources** lists every retrieved chunk in retrieval order,
one expander each. Chunks the answer actually cites (from the generator's
structured `citations`, not from searching the text) are marked **· cited**.
Each expander shows only the fields that exist: document name, page, section,
chunk id, document id, and *Distance (lower is closer)*, plus an excerpt of
up to 400 characters (turn excerpts off in the sidebar). When the answer is
the fallback message, the sources are labelled as searched but not
supporting an answer. Warnings appear when an answer has no citations, or
when the model cited a source number that was not retrieved (that citation
is removed).

The sidebar's **Chunks to retrieve** (1–10, default 3) only affects the chat.
It never changes the Stage 10 evaluation configuration.

### Feedback

Every answer has **👍 Helpful** and **👎 Not helpful** buttons with an optional
comment. A click saves once through `services/feedback_service.py`, tied to
that answer's `response_id`, and the form is replaced by a confirmation, so
the same answer cannot be rated twice in one session. If the database write
fails, an error is shown, the answer stays visible, and you can try again.
The **Feedback** page shows totals, the helpful percentage, and the latest
10/25/50 ratings (date, rating, question preview, comment, sources, response
id). It does not show answers or excerpts, and has no delete controls.

### Evaluation reports

The **Evaluation** page lists report folders from `reports/evaluation/`
(newest first) and `reports/baseline/`, opens the newest by default, and
shows the key metrics, failed or weak cases, a per-category table, and
downloads of `summary.json`, `results.csv` and `failures.jsonl`. It never
runs an evaluation and never writes to reports; run one with
`scripts/run_rag_evaluation.py` (see Stage 10). Missing or malformed reports
show a message instead of an error.

### Clear the conversation

**Clear conversation** in the sidebar removes this browser session's chat
only. Documents, ChromaDB, saved feedback and evaluation reports are kept.

### Common messages

| Message | What to do |
| --- | --- |
| *No documentation is indexed yet* | Index a document on Manage Documents. |
| *The OpenAI API key is missing* | Add `OPENAI_API_KEY` to `.env` and restart. |
| *OpenAI rejected the API key* | Check the key in `.env`. |
| *OpenAI is rate-limiting requests, or the account has no credits left* | Wait, or check billing. |
| *The request to OpenAI timed out* / *Could not reach OpenAI* | Check the network and retry. |
| *The document database (ChromaDB) could not be opened* | Check `data/chroma`; restart the app. |
| *Could not extract text from this file* | The PDF is damaged or encrypted. |
| *The text file is not UTF-8 encoded* | Re-save the file as UTF-8. |
| *Could not reach the feedback database* | Check `data/feedback.db` is not locked. |

### Security measures

- Upload type, size (also enforced by `server.maxUploadSize` in
  `.streamlit/config.toml`) and emptiness are validated before any processing.
- Uploaded files are processed in memory. The browser-supplied filename is
  sanitised for display and ids and is never used as a filesystem path.
- Document text (excerpts, filenames, sections) is shown literally or with
  Markdown escaped; raw HTML is never enabled. Image embeds in answers are
  reduced to their alt text, so a malicious document cannot make the browser
  load an external image.
- Report selection uses an allow-list of discovered folders, and resolved
  paths must stay inside the report folders, so path traversal is rejected.
  Only three report files can be downloaded.
- The ChromaDB path is fixed in code. The status panel uses a read-only
  lookup that never creates a collection.
- Errors shown to users are fixed messages. Logs record exception type names
  only, never API keys, exception text or document content.

### Local-development limitations

- No authentication. Anyone who can open the app can index documents, reset
  the collection and see the Feedback and Evaluation pages. Run it on
  localhost only.
- Chat history lives in the browser session and is lost on refresh.
- Indexing and answering run inside the Streamlit request, so a large PDF
  blocks that browser tab until it finishes.
- There is no relevance threshold yet: every retrieved chunk is listed.
- Re-indexing one document requires resetting the whole collection.

## What Stage 10 adds

**Automated RAG evaluation.** Every golden case from Stage 9 is run through the
app's real pipeline (`rag.retriever.retrieve_chunks`, then
`rag.generator.generate_answer`), and the results are scored against the
golden labels. There is no separate "evaluation pipeline": if the app changes,
the evaluation measures the change.

```text
Golden case -> retrieve top-k -> retrieval metrics
            -> generate answer -> answer, citation, grounding, abstention metrics
            -> per-case result -> summary report
```

Only the **question** (and `top_k`) is sent to the retriever and generator.
`expected_answer`, `expected_keywords`, `expected_source_ids`, `answerable` and
`notes` are used only for scoring afterwards, so the model can never see the
answer key (no evaluation-data leakage). A test checks this.

An evaluation run only **reads** ChromaDB. It never re-embeds or re-indexes
documents, never edits the golden dataset, and never opens the feedback database.

- `evaluation/config.py`: `EvaluationConfig`, thresholds, accepted fallback phrases.
- `evaluation/metrics.py`: deterministic metrics (no LLM).
- `evaluation/judge.py`: optional LLM judge.
- `evaluation/results.py`: `CaseResult` and `MetricSummary` schemas.
- `evaluation/runner.py`: `run_evaluation(dataset_path, config)`.
- `evaluation/reporter.py`: report files.
- `scripts/run_rag_evaluation.py`: the CLI.

`generate_answer` now also returns `citations` (the retrieved sources the final
answer cites, with chunk ids) and `unverified_citation_numbers` (source numbers
the model cited that were never retrieved; they are still removed from the
answer). The Streamlit display is unchanged.

### Retrieval evaluation vs answer evaluation

- **Retrieval evaluation** asks: did the search find a chunk that contains the
  answer? It needs only embeddings and ChromaDB, is cheap, and is deterministic
  for a fixed index. Run it alone with `--skip-generation`.
- **Answer evaluation** asks: given those chunks, did the model answer well,
  cite correctly, and refuse when it should? It costs one generation call per
  case.

A bad answer with a good retrieval score points at the prompt or model. A bad
retrieval score points at chunking, embeddings or `top_k`.

### Retrieval metrics (answerable cases only)

A retrieved chunk is **relevant** when its `chunk_id` exactly equals one of the
case's `expected_source_ids` (after trimming and lower-casing). A `document_id`
match also counts, for datasets that list whole documents. The filename is
compared with `expected_source_titles` only when a chunk has no stable id at
all. There is no substring matching, so `doc_p1_c1` never matches `doc_p1_c12`.
Each result records which expected id matched and which metadata fields were
missing.

- **Source hit**: 1 if any retrieved chunk is relevant, else 0.
- **Hit@k** (k = 1, 3, 5): 1 if a relevant chunk is in the first k results.
  Hit@k is `null` when fewer than k results were *requested* (`--top-k 3`
  cannot measure Hit@5). If the collection holds fewer than k chunks, every
  chunk was searched, so Hit@k is still measured.
- **MRR** (Mean Reciprocal Rank): for each case, `1 / rank` of the first
  relevant chunk (0 if none), averaged. First relevant chunk at rank 2 gives
  0.5. MRR rewards putting the right chunk *first*, which Hit@5 does not.

### Answer metrics (deterministic)

- **Keyword coverage** = matched expected keywords / total expected keywords.
  Case-insensitive, extra spaces ignored, whole words or phrases only (`with`
  does not match `without`). `null` when a case has no expected keywords.
  Matched and missing keywords are stored per case.
- **Citation precision** = citations pointing at a retrieved chunk / all
  citations the model made (including stripped `[Source N]` numbers that were
  never retrieved). Summed over all cases (micro-average). Only the structured
  `citations` are used; naming a document in the answer text earns nothing.
- **Expected-source citation hit**: the answer cites at least one expected chunk.
- **Grounding check** (conservative): has retrieved context, has citations,
  citations all refer to retrieved chunks, fallback used, and whether an
  answer was given without useful evidence. `grounded` is true only for a
  non-fallback answer that cites retrieved chunks and nothing else. This is
  citation-level grounding: it does **not** prove every sentence is supported.

### Abstention evaluation

For `answerable: false` cases the right behaviour is the fallback message.
An answer **abstained** when it contains an accepted fallback phrase
(`DEFAULT_FALLBACK_PHRASES` in `evaluation/config.py`; case, punctuation and
spacing are ignored). It is a **correct abstention** only if it also cites no
chunks, so the model did not present retrieved text as evidence. Retrieval
hit metrics are not computed for these cases (there is nothing to find), but
the retrieved chunks are still stored for inspection.

For answerable cases, abstaining is counted as an **incorrect abstention**.

### Limitations of deterministic metrics

- Keyword coverage checks wording, not truth: "One lap" misses the keyword
  `1 lap`, and an answer can contain every keyword and still be wrong.
- A citation to a retrieved chunk does not prove that chunk supports the claim.
- Relevance is only as good as `expected_source_ids`. If a fact also appears in
  a chunk nobody listed, retrieving that chunk counts as a miss. Chunk ids also
  change when a document is re-chunked, so re-check the dataset after re-indexing.

### Optional LLM judge

`--use-llm-judge` adds a second model call per case that scores the answer
against the reference answer and the retrieved context:

| Field | Scale |
| --- | --- |
| `correctness` | 0 wrong, 1 mostly wrong, 2 partly right, 3 mostly right, 4 fully right |
| `grounding` | 0 unsupported ... 4 every claim supported by the context |
| `relevance` | 0 off-topic ... 4 directly answers the question |
| `unsupported_claims` | `true` if any claim is not in the context |
| `reason` | short explanation |

It uses the project's OpenAI client, structured output validated with Pydantic
(`JudgeVerdict`), temperature 0 (retried without it if a model rejects it), and
`gpt-5.4-mini` by default (`--judge-model` to change). The model and temperature
are written to `summary.json`.

It is **off by default** because it doubles the API cost and can vary between
runs. Judge scores are model-based estimates, not objective truth: in testing,
the judge gave a wrong refusal a relevance of 4 despite being told not to.
A judge failure is recorded on that case (`judge_error`) and the run continues.
Judge scores never change the deterministic metrics. By default the judge is
the same model as the generator, which may be lenient toward its own answers.

### Run an evaluation

Full evaluation (retrieval + answers, about one embedding call and one
generation call per case):

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --dataset data\evaluation\golden_dataset.jsonl --top-k 5
```

Retrieval only (embedding calls only, no generation):

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --skip-generation
```

Filter cases (filters combine; order is case ids, category, difficulty, max):

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --category race_distances --max-cases 3
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --difficulty advanced
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --case-id eval_006 --case-id eval_019
```

An unknown case id, a filter that matches nothing, an empty dataset, or
`--top-k 0` stops with a clear message. Add `-v` to see progress logs.

From Python:

```python
from evaluation.config import EvaluationConfig
from evaluation.runner import run_evaluation

run = run_evaluation("data/evaluation/golden_dataset.jsonl",
                     EvaluationConfig(top_k=5, generate_answers=False))
print(run.summary["metrics"]["mrr"], run.report_dir)
```

`run_evaluation` also accepts `retrieve=`, `generate=` and `judge=` callables,
which the tests use to run offline with fakes.

### Thresholds and exit codes

No threshold is checked unless you ask for one:

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --fail-below-source-hit 0.8 --fail-below-mrr 0.6 --fail-below-keyword-coverage 0.6 --fail-below-abstention 0.8
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --recommended-thresholds
```

`--recommended-thresholds` applies the suggested starting points in
`RECOMMENDED_THRESHOLDS` (`evaluation/config.py`); explicit `--fail-below-*`
flags override them. They are starting points for this small dataset, not
universal targets. A metric that could not be measured (for example keyword
coverage with `--skip-generation`) fails its threshold.

| Exit code | Meaning |
| --- | --- |
| `0` | PASS: all configured thresholds met and case errors at most `--max-case-errors` |
| `1` | FAIL: a threshold was missed or too many cases raised errors (reports are still written) |
| `2` | Could not run: invalid configuration, dataset not loadable, or ChromaDB collection missing or empty (no reports written) |

**Case-error policy:** an error in one case (for example an API timeout) is
stored on that case and the run continues. By default any case error makes the
run FAIL (`--max-case-errors 0`), because a metric computed over fewer cases is
not comparable with the baseline. Raise the limit to tolerate flaky cases.

### Reports

Each run writes a new folder `reports/evaluation/YYYY-MM-DD_HHMMSS/` (a suffix
such as `_2` is added if the name is taken, so earlier runs are never
overwritten). `--output-dir` changes the parent folder; `--run-dir PATH` uses an
exact folder, which must be new or empty.

| File | Contents |
| --- | --- |
| `summary.json` | timestamp, dataset path and counts, config, models, collection, metrics, breakdowns by category / difficulty / answerability, errors, thresholds, PASS/FAIL, package versions and git commit |
| `results.jsonl` | one complete `CaseResult` per case: retrieved chunks with match details, answer, citations, every metric, latencies, errors |
| `results.csv` | the key flat fields, for a spreadsheet |
| `failures.jsonl` | cases with `failure_reasons`: `error`, `missed_expected_sources`, `low_keyword_coverage` (below 0.5), `incorrect_citations`, `failed_abstention`, `incorrect_abstention` |

Reports contain questions, answers and chunk ids, but no API keys, prompts or
environment variables. `reports/evaluation/` is git-ignored; the baseline in
`reports/baseline/` is meant to be committed.

### Baseline (2026-09-28)

The first run, on the existing system with no tuning:

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_evaluation.py --top-k 5 --run-dir reports\baseline --recommended-thresholds
```

Configuration: `top_k` 5, `text-embedding-3-small`, collection
`developer_docs` (13 chunks of `Speed Schedule (1).pdf`), generator
`gpt-5.4-mini`, no judge, git commit `beb884c`. The app's own default is
`top_k` 3. Hit@3 shows how retrieval does at that setting (the ranking is
the same), but the answers here were generated from 5 chunks.

| Metric | Value |
| --- | --- |
| Cases (answerable / unanswerable) | 23 (18 / 5), 0 errors |
| Source-hit rate (= Hit@5) | 1.000 |
| Hit@1 / Hit@3 | 0.500 / 0.667 |
| MRR | 0.640 |
| Keyword coverage | 0.799 |
| Citation precision | 1.000 |
| Expected-source citation hit | 0.833 |
| Correct abstention rate | 1.000 (5 / 5) |
| Incorrect abstentions | 1 (`eval_006`) |
| Avg retrieval / total latency | 474 ms / 1966 ms |

Weak cases: `eval_006` refused although the right chunk was ranked first (the
answer needs 3 x 200m = 600m, combining two facts); `eval_005` and `eval_007`
gave correct but terse answers (low keyword coverage, and "One lap" does not
match the keyword `1 lap`); `race_distances` questions have the lowest MRR
(0.24), with the relevant table chunks often at rank 4 or 5.

### Compare with the baseline

After a change (chunking, `top_k`, prompt, model), run the same command into a
new folder and compare `metrics` in the two `summary.json` files, then the
per-case rows in `results.csv`. Change one thing at a time, keep the dataset
unchanged between the runs you compare, and remember that answer metrics can
move a little between identical runs because generation is not fully
deterministic.

## What Stage 9 adds

A **golden evaluation dataset**: a version-controlled set of questions with the
answer and source chunks a good response should have. Stage 9 only builds and
validates the dataset. It does **not** score answers or run the RAG pipeline;
that is the next stage.

- `data/evaluation/golden_dataset.jsonl`: one JSON case per line (committed).
- `evaluation/schemas.py`: the Pydantic model `EvaluationCase`.
- `evaluation/dataset.py`: `load_evaluation_dataset(path)` and helpers.
- `scripts/validate_evaluation_dataset.py`: validator CLI (exit code 0 or 1).
- `scripts/export_feedback_candidates.py`: turns thumbs-down feedback into
  *unreviewed* candidates (never into golden cases).

Loading and validating never call an LLM or the embedding API.

### Starter dataset

The starter cases are written against the document that is currently indexed:
`Speed Schedule (1).pdf` (`document_id` `speed_schedule_1`, 13 chunks), a
roller-skating championship schedule. They are **not** Python documentation
questions. If you index other documents, add cases for them, and re-check the
existing ones, because chunk ids change when a document is re-chunked.

### JSONL fields

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | unique, stable id such as `eval_001`; never reuse or renumber |
| `question` | yes | the user question |
| `expected_answer` | yes | short reference answer (for unanswerable cases: the safe behaviour) |
| `expected_keywords` | yes | concepts a good answer should mention; may be `[]` |
| `expected_source_ids` | yes | retrieval ids that support the answer; `[]` when unanswerable |
| `category` | yes | subject area, e.g. `reporting_rules` |
| `difficulty` | yes | `beginner`, `intermediate` or `advanced` |
| `answerable` | yes | `true` if the indexed documents contain the answer (real JSON boolean) |
| `expected_source_titles` | no | human-readable document names |
| `notes` | no | why the case exists, traps, caveats |
| `tags` | no | free labels such as `multi_part`, `similar_sections`, `rewording` |
| `metadata` | no | free-form object; the starter set stores `document_id` and `pages` |

Rules enforced by the loader:

- `id`, `question`, `expected_answer` and `category` must not be empty.
- IDs must be unique across the file.
- Unknown fields are rejected, so a typo like `expected_keyword` is caught.
- `null` lists become `[]`.
- An answerable case needs at least one `expected_source_ids` entry.
- Every problem is reported with its line number; nothing is skipped silently.

`expected_source_ids` uses **chunk ids** (for example `speed_schedule_1_p1_c2`),
the same ids shown as *Chunk ID* in the app. When a fact is repeated (like the
reporting-time rule), every chunk containing it is listed, and retrieving *any
one* of them counts as finding the source (Stage 10 scores it this way).

Example case (source list shortened):

```json
{"id": "eval_005", "question": "How long is one lap according to the schedule?", "expected_answer": "One lap is the length of a 200m rink, as per RSFI guidelines.", "expected_keywords": ["1 lap", "200m", "rink", "RSFI"], "expected_source_ids": ["speed_schedule_1_p1_c2", "speed_schedule_1_p2_c5"], "expected_source_titles": ["Speed Schedule (1).pdf"], "category": "race_format", "difficulty": "beginner", "answerable": true, "notes": "", "tags": ["direct_fact"], "metadata": {"document_id": "speed_schedule_1", "pages": [1, 2]}}
```

### Unanswerable cases

Set `"answerable": false` and `"expected_source_ids": []` for questions the
indexed documents cannot answer: out-of-scope questions, or details the
document never states (a venue, a fee, results, an unexplained abbreviation).
The `expected_answer` describes the safe behaviour: the assistant says the
uploaded documents do not contain the information and does not guess. In this
project that means the fixed fallback message. Retrieval will still return the
nearest chunks for such questions, and that is expected.

### How to add a reviewed evaluation case

1. Ask the question in the app, or look through the chunks in the upload view,
   and find the chunk(s) that really support the answer. Copy their *Chunk ID*
   values exactly.
2. Write a short `expected_answer` in your own words (do not paste long passages).
3. Append **one line** to `golden_dataset.jsonl` with the next free `id`.
4. Validate (below). Use `--check-index` to confirm the chunk ids exist.
5. Commit the dataset together with the document it refers to, or note which
   document must be indexed.

### Validate the dataset

```powershell
.\.venv\Scripts\python.exe scripts\validate_evaluation_dataset.py
.\.venv\Scripts\python.exe scripts\validate_evaluation_dataset.py --dataset data\evaluation\golden_dataset.jsonl
.\.venv\Scripts\python.exe scripts\validate_evaluation_dataset.py --check-index
```

It prints the path, totals, answerable/unanswerable counts, counts per category
and difficulty, any invalid or duplicate records, and a final `PASS` or `FAIL`.
The exit code is `0` on PASS and `1` on FAIL, so it can run in CI.
`--check-index` also confirms each `expected_source_ids` entry exists in the
local ChromaDB collection (chunk ids only, no API calls). Leave it off in CI
where no index exists.

### Export feedback candidates

Thumbs-down feedback (Stage 8) can point at questions worth testing:

```powershell
.\.venv\Scripts\python.exe scripts\export_feedback_candidates.py --output data\evaluation\feedback_candidates.jsonl
```

- The feedback database is opened **read-only**; no feedback is changed.
- Each distinct question (compared after lower-casing and collapsing spaces)
  becomes one candidate with `"review_status": "unreviewed"`, the user's
  comments, and the retrieved chunk ids. It has **no expected answer**, so it
  is not a valid golden case yet.
- Not exported: session ids, answers, source excerpts, and model details.
- Re-running only appends new questions. Questions already exported, or already
  in the golden dataset, are skipped, and your manual edits are kept.
- The output file is git-ignored because it contains user questions and comments.

> **Warning:** feedback candidates need human review. A thumbs-down can mean a
> bad answer, a bad retrieval, or just a user who disliked the reply. Decide
> what the correct answer is, find the supporting chunks yourself, then write a
> proper case by hand. Never copy candidates into the golden dataset unchecked.

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

Since Stage 11 the interface uses **👍 Helpful / 👎 Not helpful** buttons and
accepts one rating per answer per session (see Stage 11). The service still
upserts, so a row is never duplicated.

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
or page number cannot be hallucinated. Since Stage 11 each source is an
expander labelled like `[1] speed_schedule.pdf — page 3 · cited`. When the answer is the fallback message,
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

The app runs at http://localhost:8501. Use the sidebar to move between
**Ask Documentation**, **Manage Documents**, **Feedback** and **Evaluation**.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -p "test_*.py" -v
```

Or a single module:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_generator -v
```

Stage 10 evaluation tests only (fake retriever and generator, no API calls):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_evaluation_metrics tests.test_evaluation_runner tests.test_evaluation_reporter tests.test_evaluation_cli -v
```

Stage 11 interface tests only (fakes and temporary folders, no API calls):

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ui_helpers tests.test_ui_services tests.test_ui_reports tests.test_ui_app -v
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
├── evaluation/
│   ├── schemas.py           EvaluationCase (Pydantic)
│   ├── dataset.py           load + validate the JSONL dataset
│   ├── candidates.py        feedback -> unreviewed candidates
│   ├── config.py            EvaluationConfig, thresholds, fallback phrases
│   ├── metrics.py           deterministic retrieval + answer metrics
│   ├── judge.py             optional LLM judge
│   ├── results.py           CaseResult, MetricSummary
│   ├── runner.py            run_evaluation()
│   └── reporter.py          summary.json, results.jsonl/.csv, failures.jsonl
├── services/
│   └── feedback_service.py  SQLite feedback store (upsert, summary, listing)
├── scripts/
│   ├── view_feedback.py     developer viewer for saved feedback
│   ├── validate_evaluation_dataset.py
│   ├── export_feedback_candidates.py
│   └── run_rag_evaluation.py  Stage 10 evaluation CLI
├── tests/
│   ├── evaluation_fakes.py   fake retriever/generator for offline tests
│   ├── test_evaluation_cli.py
│   ├── test_evaluation_dataset.py
│   ├── test_evaluation_metrics.py
│   ├── test_evaluation_reporter.py
│   ├── test_evaluation_runner.py
│   ├── test_ui_app.py        Streamlit AppTest smoke tests
│   ├── test_ui_helpers.py
│   ├── test_ui_reports.py
│   ├── test_ui_services.py
│   ├── test_feedback_candidates.py
│   ├── test_feedback_service.py
│   ├── test_generator.py
│   └── test_ingestion.py
├── data/chroma/          persistent ChromaDB collection (git-ignored)
├── data/feedback.db      feedback ratings (git-ignored)
├── data/evaluation/      golden_dataset.jsonl (committed)
├── reports/baseline/     Stage 10 baseline report
├── reports/evaluation/   timestamped evaluation runs (git-ignored)
├── ui/
│   ├── config.py         UI limits and defaults
│   ├── services.py       boundary to rag/services/evaluation, safe errors
│   ├── state.py          chat and feedback session state
│   ├── validation.py     question/upload validation, filename sanitising
│   ├── formatting.py     source display, excerpts, safe Markdown
│   ├── reports.py        read-only evaluation report access
│   ├── components.py     shared widgets and sidebar
│   └── pages/            ask.py, documents.py, feedback.py, evaluation.py
├── .streamlit/config.toml  upload size limit
├── app.py                Streamlit entry point and navigation
├── .env                  OPENAI_API_KEY (git-ignored)
└── requirements.txt
```

## Models

| Purpose | Model | Set in |
| --- | --- | --- |
| Embeddings | `text-embedding-3-small` | `rag/embeddings.py` (`MODEL_NAME`) |
| Answer generation | `gpt-5.4-mini` | `rag/generator.py` (`MODEL_NAME`) |
| Evaluation judge (optional) | `gpt-5.4-mini` | `evaluation/judge.py` (`DEFAULT_JUDGE_MODEL`), or `--judge-model` |

## Known limitations

- `section` is detected from Markdown headings only, so PDF chunks usually have
  no section name.
- Chunking is character-based and splits at fixed offsets, so a chunk can begin
  mid-sentence.
- Feedback is stored per answer, not per source, so a rating cannot say which
  cited source was wrong. There is no login, so `session_id` only identifies a
  browser session, and anyone who can open the app can rate.
- Every retrieved chunk is listed as a source, including ones the answer did not
  cite (cited ones are marked). There is no relevance threshold yet.
- The interface has no authentication and is meant for localhost only.
