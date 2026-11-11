"""Hybrid retrieval: keyword + vector candidates, re-ranked.

The vector side is TF-IDF built from scratch (no embedding provider here):
each memory is a document, terms are weighted by inverse document frequency,
and similarity is cosine similarity over TF-IDF vectors. The keyword side is
an IDF-weighted term-overlap score. Re-ranking combines relevance, recency
and source reliability. The production path swaps the TF-IDF vectors for a
real embedding provider (pgvector) behind the same interface.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .schema import SOURCE_RELIABILITY, Memory

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no retrieval signal and would otherwise dominate
# short queries (a query sharing only "what"/"is" with a document is not
# evidence about that document).
STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "what", "which", "who", "whom", "whose",
        "when", "where", "why", "how", "of", "to", "in", "on", "for",
        "with", "by", "at", "from", "as", "and", "or", "but", "not", "no",
        "my", "your", "his", "her", "its", "our", "their", "this", "that",
        "these", "those", "it", "s", "am", "i", "you", "he", "she", "we",
        "they", "me", "him", "us", "them", "can", "could", "should",
        "would", "will", "shall", "may", "might", "must", "have", "has",
        "had", "if", "then", "than", "so", "such", "own", "just", "about",
        "into", "over", "after", "before", "between", "through", "during",
        "under", "again", "further", "once", "here", "there", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "only",
        "same", "too", "very",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    keyword_score: float
    vector_score: float
    recency_score: float
    conflict: bool = False
    superseded_versions: list = None

    def __post_init__(self) -> None:
        if self.superseded_versions is None:
            self.superseded_versions = []


class TfidfIndex:
    """Minimal in-memory TF-IDF index over memory documents.

    Documents are memory ids; terms come from content plus the memory key.
    IDF is computed live from the current document set so deletions are
    reflected immediately (deletion completeness).
    """

    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = {}

    def add(self, memory_id: str, text: str) -> None:
        self._docs[memory_id] = tokenize(text)

    def remove(self, memory_id: str) -> None:
        self._docs.pop(memory_id, None)

    def __len__(self) -> int:
        return len(self._docs)

    def contains(self, memory_id: str) -> bool:
        return memory_id in self._docs

    def _idf(self) -> dict[str, float]:
        n = len(self._docs)
        df: dict[str, int] = {}
        for tokens in self._docs.values():
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    def _vector(self, tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        tf: dict[str, float] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0.0) + 1.0
        vec = {}
        for term, count in tf.items():
            if term in idf:
                vec[term] = (1.0 + math.log(count)) * idf[term]
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return {}
        return {term: value / norm for term, value in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        return sum(value * large.get(term, 0.0) for term, value in small.items())

    def search(self, query: str, memory_ids: set[str] | None = None) -> dict[str, dict]:
        """Return per-memory keyword and vector scores for the query.

        Only ids in ``memory_ids`` (when given) are scored, so the store can
        restrict retrieval to one tenant's active memories.

        Query terms absent from every document get the maximum IDF: their
        absence from a document is evidence *against* it, so a query that
        only shares a low-information term (e.g. a name) with a document
        scores near zero instead of matching on that term alone.
        """
        idf = self._idf()
        default_idf = math.log(len(self._docs) + 1) + 1.0
        query_tokens = tokenize(query)
        query_terms = set(query_tokens)
        query_idf = dict(idf)
        for term in query_terms:
            query_idf.setdefault(term, default_idf)
        query_vec = self._vector(query_tokens, query_idf)
        results: dict[str, dict] = {}
        for memory_id, tokens in self._docs.items():
            if memory_ids is not None and memory_id not in memory_ids:
                continue
            doc_vec = self._vector(tokens, idf)
            vector_score = self._cosine(query_vec, doc_vec)
            doc_terms = set(tokens)
            overlap = query_terms & doc_terms
            if overlap and query_terms:
                keyword_score = sum(query_idf[t] for t in overlap) / sum(
                    query_idf[t] for t in query_terms
                )
            else:
                keyword_score = 0.0
            results[memory_id] = {
                "keyword_score": keyword_score,
                "vector_score": vector_score,
            }
        return results


def recency_score(recorded_at: datetime, now: datetime | None = None) -> float:
    """Exponential decay with a 180-day half-life; fresh memories score ~1."""
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - recorded_at).total_seconds() / 86400.0)
    return 0.5 ** (age_days / 180.0)


def reliability(memory: Memory) -> float:
    if memory.origin == "inferred":
        return SOURCE_RELIABILITY["inference"]
    return SOURCE_RELIABILITY.get(memory.source_type, 0.8)


def rerank(
    memory: Memory,
    keyword_score: float,
    vector_score: float,
    now: datetime | None = None,
    weights: dict | None = None,
) -> float:
    """Final score: relevance, recency and source reliability.

    Weights default to relevance 0.55, recency 0.25, reliability 0.20.
    """
    weights = weights or {"relevance": 0.55, "recency": 0.25, "reliability": 0.20}
    relevance = 0.5 * keyword_score + 0.5 * vector_score
    rec = recency_score(memory.recorded_at, now)
    rel = reliability(memory)
    return (
        weights["relevance"] * relevance
        + weights["recency"] * rec
        + weights["reliability"] * rel
    )
