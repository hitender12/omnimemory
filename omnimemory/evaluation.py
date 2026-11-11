"""Evaluation suite for OMNIMEMORY retrieval and memory updates.

Metrics (all computed against explicit fixtures, no hidden state):

* recall@K / precision@K — standard retrieval quality.
* citation accuracy — fraction of returned results whose source_id matches
  the expected evidence for the query.
* stale-memory rate — fraction of returned results that are past valid_to.
* contradiction resolution — a contradictory pair must both be preserved,
  the older marked superseded + contradicted, the newer current.
* deletion completeness — after DELETE, the fact is gone from search results
  and from the vector index.
* tenant isolation — a search in tenant B returns nothing from tenant A.
"""

from __future__ import annotations

from .retrieval import ScoredMemory
from .store import MemoryStore


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved = retrieved_ids[:k]
    return len(set(retrieved) & relevant_ids) / len(retrieved) if retrieved else 0.0


def citation_accuracy(
    results: list[ScoredMemory], expected_source_ids: set[str]
) -> float:
    """Fraction of returned results whose source is an expected source."""
    if not results:
        return 0.0
    good = sum(1 for r in results if r.memory.source_id in expected_source_ids)
    return good / len(results)


def stale_memory_rate(results: list[ScoredMemory], now=None) -> float:
    """Fraction of returned results that are stale (past valid_to)."""
    if not results:
        return 0.0
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    stale = sum(
        1
        for r in results
        if r.memory.valid_to is not None and r.memory.valid_to <= now
    )
    return stale / len(results)


def contradiction_resolution_check(store: MemoryStore, key: str, tenant_id: str) -> dict:
    """Both versions of a contradictory fact must be preserved and marked."""
    versions = [
        m
        for m in store._memories.values()
        if m.tenant_id == tenant_id and m.key == key
    ]
    current = [m for m in versions if m.status == "active"]
    superseded = [m for m in versions if m.status == "superseded"]
    contradicted = [m for m in versions if m.contradicted]
    passed = (
        len(versions) >= 2
        and len(current) == 1
        and len(superseded) >= 1
        and len(contradicted) >= 1
        and superseded[0].superseded_by == current[0].id
    )
    return {
        "passed": passed,
        "versions": len(versions),
        "current": len(current),
        "superseded": len(superseded),
        "contradicted": len(contradicted),
    }


def deletion_completeness_check(
    store: MemoryStore, tenant_id: str, deleted_id: str, query: str
) -> dict:
    """A deleted fact must be absent from search results and from the index."""
    results = store.search(query, tenant_id, k=10)
    in_results = any(r.memory.id == deleted_id for r in results)
    in_index = store.index_contains(deleted_id)
    return {
        "passed": (not in_results) and (not in_index),
        "in_search_results": in_results,
        "in_index": in_index,
    }


def tenant_isolation_check(
    store: MemoryStore, tenant_a: str, tenant_b: str, query: str
) -> dict:
    """Searching as tenant B must never surface tenant A's memories."""
    results = store.search(query, tenant_b, k=10)
    leaked = [r.memory.id for r in results if r.memory.tenant_id == tenant_a]
    return {"passed": not leaked, "leaked_ids": leaked}


def evaluate(store: MemoryStore, cases: list[dict], k: int = 5) -> dict:
    """Run a fixture of retrieval cases and aggregate the metrics.

    Each case: {"query", "tenant_id", "relevant_ids", "expected_sources"}.
    """
    recalls, precisions, citations = [], [], []
    per_case = []
    for case in cases:
        results = store.search(
            case["query"], case["tenant_id"], k=k, include_inferred=True
        )
        ids = [r.memory.id for r in results]
        relevant = set(case.get("relevant_ids", []))
        r = recall_at_k(ids, relevant, k)
        p = precision_at_k(ids, relevant, k)
        c = citation_accuracy(results, set(case.get("expected_sources", [])))
        recalls.append(r)
        precisions.append(p)
        citations.append(c)
        per_case.append(
            {
                "query": case["query"],
                "recall@k": round(r, 3),
                "precision@k": round(p, 3),
                "citation_accuracy": round(c, 3),
            }
        )
    n = max(1, len(cases))
    return {
        "cases": len(cases),
        "k": k,
        "mean_recall@k": round(sum(recalls) / n, 3),
        "mean_precision@k": round(sum(precisions) / n, 3),
        "mean_citation_accuracy": round(sum(citations) / n, 3),
        "per_case": per_case,
    }
