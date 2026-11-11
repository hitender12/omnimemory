# OMNIMEMORY — Architecture

## Design principles

1. **Memory is user-controlled.** Inspect, correct, export, delete — all first-class operations, not afterthoughts.
2. **History is never destroyed by an update.** A correction creates a new version; the old version is marked superseded and stays queryable.
3. **Retrieval must be evidence-backed.** Every result carries source IDs, timestamps, confidence and scores. If the evidence is missing or contradictory, the answer is "I don't know".
4. **Inferences are second-class.** An inferred fact is stored with `origin="inferred"`, excluded from default retrieval, and never presented as something the user said.
5. **Tenants are isolated at every layer.** Store operations are tenant-scoped; cross-tenant access is a hard error (403), not an empty result.

## Components

```text
client (SDK) ──HTTP──> server (stdlib http.server)
                            │
                     MemoryStore (facts, entities, relations)
                            ├── TfidfIndex (keyword + vector candidates)
                            ├── rerank (relevance / recency / reliability)
                            └── evaluation (recall, precision, citation, ...)
```

### `schema.py`

`Memory`, `Entity`, `Relation` dataclasses with validation. A memory has full
provenance (`source_type`, `source_id`, `source_excerpt`), temporal fields
(`valid_from`, `valid_to`, `recorded_at`, `superseded_by`), `status`
(active / superseded / expired / deleted), `contradicted`, `confidence`,
`extractor_version`, `origin`, `sensitivity_class`, `consent_basis` and
`retention_days`. A `key` field (e.g. `preference:coffee`) identifies the
attribute a fact is about, which is what makes contradiction detection
possible without an LLM.

### `store.py` — `MemoryStore`

- `add()` — validates, handles key conflicts (`supersede` by default:
  the older version is marked superseded + contradicted; `keep_both`
  leaves both active and flagged), indexes the new fact.
- `search()` — tenant-scoped candidates (active only, unless
  `include_history`; inferred excluded unless requested; expired
  excluded), TF-IDF scores from the index, re-rank, top-k. Memories with
  zero term overlap are never returned. Live contradictions (same key,
  multiple active versions, different content) are flagged on results.
- `answer()` — returns the top answer with evidence, or "I don't know"
  when nothing relevant matches (relevance floor 0.30), or when the top
  match is contradictory.
- `correct()` — creates a new version; the old one is superseded with
  `superseded_by` pointing at the new version. `history()` walks the chain.
- `delete()` — removes the fact from the store *and* the index.
  Superseded versions stay indexed so history search can find them;
  only delete/expire removes a document from the index.
- `purge_expired()` — marks memories past `valid_to` or their retention
  deadline as expired.
- `export()` / `forget_tenant()` — full tenant export (JSON) and complete
  deletion across facts, entities, relations and index.
- Entities/relations — people, projects, organizations with typed edges.

### `retrieval.py` — `TfidfIndex` + re-rank

TF-IDF built from scratch: tokenize (lowercase alphanumerics, stopwords
removed), IDF computed live from the current document set (so deletions
are reflected immediately), cosine similarity over L2-normalized vectors.
Two scoring details matter:

- **OOV query terms get maximum IDF.** A term appearing in zero documents
  is maximally discriminative; its absence from a document counts against
  that document. Without this, a query sharing only a common name with a
  document scores as a strong match.
- **Zero-overlap filtering.** A candidate with keyword and vector score
  both zero is dropped before re-ranking, so recency and source
  reliability can never manufacture an answer from nothing.

Re-rank: `0.55 * relevance + 0.25 * recency + 0.20 * reliability`, where
relevance is the mean of keyword and vector scores, recency is exponential
decay (180-day half-life), and reliability is 1.0 for user-stated facts,
0.95 conversation, 0.85 document, 0.5 inferred.

### `evaluation.py`

Pure functions over the store: `recall_at_k`, `precision_at_k`,
`citation_accuracy`, `stale_memory_rate`,
`contradiction_resolution_check`, `deletion_completeness_check`,
`tenant_isolation_check`, and `evaluate()` which aggregates a fixture of
query cases into mean recall/precision/citation accuracy.

### `server.py` / `client.py`

A `ThreadingHTTPServer` wrapper (stdlib only) exposing the full API, and a
thin `urllib`-based SDK client. Tenant isolation is enforced per request.

## Data model (production)

```text
memories(id, tenant_id, subject_id, type, key, content, origin,
         source_type, source_id, source_excerpt,
         valid_from, valid_to, recorded_at, superseded_by, status,
         confidence, extractor_version, sensitivity_class,
         consent_basis, retention_days, embedding_id)
entities(id, tenant_id, type, name, attributes)
relations(id, tenant_id, from_entity_id, relation, to_entity_id)
```

## Production path (not implemented here)

| Layer | Reference (this repo) | Production |
|---|---|---|
| API | stdlib `http.server` | FastAPI + Pydantic |
| Store | in-memory dicts | PostgreSQL, row-level tenant isolation |
| Vectors | TF-IDF from scratch | pgvector + versioned embedding provider |
| Extraction | manual `add()` | Redis queue + model worker, dedup + validation |
| Indexer | in-process | Rust `memory-event-indexer` when volume justifies it |
| Privacy | API-layer isolation | encryption at rest/in transit, backup/deletion policy |

## Known limitations

- In-memory only: no persistence, no recovery, single process.
- No authentication/authorization beyond tenant scoping; no rate limiting.
- Relevance floor (0.30) and re-rank weights are hand-tuned on the demo corpus.
- Contradiction detection is key-based (exact `key` match), not semantic: two
  differently-worded facts about the same attribute with different keys are
  not detected as contradictory.
- The HTTP server is a reference wrapper, not hardened for production use.
