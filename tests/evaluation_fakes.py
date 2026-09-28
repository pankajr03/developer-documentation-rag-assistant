"""Fake retriever/generator and dataset helpers for the Stage 10 tests.

Nothing here calls OpenAI or opens ChromaDB.
"""

import json
from pathlib import Path
from typing import Any, Optional

from rag.generator import FALLBACK_MESSAGE, build_citations, build_sources

CASES = [
    {"id": "eval_001", "question": "What is the reporting time?",
     "expected_answer": "30 minutes before race time.",
     "expected_keywords": ["30 minutes", "race time"],
     "expected_source_ids": ["doc_p1_c1"], "category": "rules",
     "difficulty": "beginner", "answerable": True},
    {"id": "eval_002", "question": "How long is one lap?",
     "expected_answer": "200m.", "expected_keywords": ["200m", "RSFI"],
     "expected_source_ids": ["doc_p2_c3"], "category": "format",
     "difficulty": "intermediate", "answerable": True},
    {"id": "eval_003", "question": "Where is the venue?",
     "expected_answer": "Not in the documents.", "expected_keywords": [],
     "expected_source_ids": [], "category": "rules",
     "difficulty": "advanced", "answerable": False},
]


def write_dataset(path: Path, cases: Optional[list[dict]] = None) -> Path:
    path.write_text(
        "\n".join(json.dumps(case) for case in (cases or CASES)) + "\n",
        encoding="utf-8")
    return path


def make_chunk(chunk_id: str, rank: int) -> dict[str, Any]:
    return {"text": f"text of {chunk_id}", "source": "Doc.pdf",
            "document_id": "doc", "chunk_id": chunk_id, "chunk_index": rank,
            "page": 1, "section": None, "distance": 0.1 * rank}


# Retrieved chunk ids per question.
RANKINGS = {
    "What is the reporting time?": ["doc_p1_c1", "doc_p2_c3", "doc_p3_c5"],
    "How long is one lap?": ["doc_p1_c1", "doc_p3_c5", "doc_p2_c3"],
    "Where is the venue?": ["doc_p3_c5", "doc_p1_c1", "doc_p2_c3"],
}

# Model answers per question.
ANSWERS = {
    "What is the reporting time?": "Report 30 minutes before race time [Source 1].",
    "How long is one lap?": "One lap is 200m [Source 3].",
    "Where is the venue?": FALLBACK_MESSAGE,
}


class FakeRetriever:
    """Records every call so tests can check no golden labels were sent."""

    def __init__(self, fail_on: tuple[str, ...] = ()):
        self.calls: list[tuple[tuple, dict]] = []
        self.fail_on = fail_on

    def __call__(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        self.calls.append(((question,), {"top_k": top_k}))
        if question in self.fail_on:
            raise RuntimeError("retrieval exploded")
        return [make_chunk(chunk_id, rank) for rank, chunk_id
                in enumerate(RANKINGS[question][:top_k], start=1)]


class FakeGenerator:
    """Mimics generate_answer's result shape without calling a model."""

    def __init__(self, answers: Optional[dict[str, str]] = None,
                 fail_on: tuple[str, ...] = ()):
        self.calls: list[tuple[tuple, dict]] = []
        self.answers = answers or ANSWERS
        self.fail_on = fail_on

    def __call__(self, question, retrieved_chunks, top_k=None):
        self.calls.append(((question, retrieved_chunks), {"top_k": top_k}))
        if question in self.fail_on:
            raise RuntimeError("generation exploded")
        sources = build_sources(retrieved_chunks)
        answer = self.answers[question]
        return {"response_id": "fake-id", "question": question,
                "answer": answer, "sources": sources,
                "citations": build_citations(answer, sources),
                "unverified_citation_numbers": [],
                "retrieved_chunks": retrieved_chunks,
                "model_name": "fake-model", "retrieval_top_k": top_k}
