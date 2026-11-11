# OMNIMEMORY

> OMNIMEMORY is a user-controlled memory API for AI apps: durable facts, temporal updates, hybrid retrieval and source-backed results.

Give AI applications durable, inspectable memory with provenance, time awareness and user control. Every retrieved fact carries its source, timestamps and confidence — and the system says **"I don't know"** when memory is missing or conflicting, instead of guessing.

## The problem

AI assistants forget. The usual fixes are worse than the disease: opaque vector stores with no provenance, silent overwrites when a fact changes, and inferred guesses presented as things the user actually said. Users cannot inspect, correct, export or delete what an app "remembers" about them.

## What this is

A memory service/API that accepts conversation events or application facts, stores structured facts with full provenance, and retrieves relevant information with source attribution.

**Differentiators**

- **Inspectable and editable** — every memory has an id, source, timestamps, confidence and a deletion path.
- **Temporal by design** — corrections supersede, never silently overwrite. History is preserved and queryable.
- **Honest retrieval** — contradictions are preserved and flagged; the system answers "I don't know" when memory is missing or conflicting.
- **User-stated vs inferred** — an inference is never presented as a confirmed user fact.
- **User control** — export everything, forget a single fact, or forget a whole tenant, with deletion verified across the fact store and the vector index.

## Quickstart

```bash
python demo.py            # scripted end-to-end lifecycle, no server needed
python -m unittest discover -s tests   # 18 tests

# HTTP server (stdlib only, port 8080 by default)
python -m omnimemory.server 8080
```

```python
from omnimemory import MemoryStore

store = MemoryStore()
store.add("acme", "user-1", "preference", "Maya prefers black coffee.",
          source_type="user", source_id="chat:120", key="preference:coffee")
store.add("acme", "user-1", "fact", "Maya leads Project Northstar.",
          source_type="document", source_id="doc:77", key="fact:project")

store.search("What coffee does Maya like?", "acme")
store.answer("What project is Maya leading?", "acme")
# -> {"status": "answered", "answer": "Maya leads Project Northstar.", "evidence": [...]}

store.answer("What is Maya's favourite film?", "acme")
# -> {"status": "unknown", "answer": "I don't know", ...}
```

Or use the SDK client against the HTTP server:

```python
from omnimemory import OmniMemoryClient

client = OmniMemoryClient("http://127.0.0.1:8080", tenant_id="acme")
memory = client.add_memory(subject_id="user-1", memory_type="fact",
                           content="Maya likes tea.", source_type="user",
                           source_id="chat:1", key="fact:drink")
client.search("tea")
client.correct_memory(memory["id"], "Maya likes coffee.")
client.delete_memory(memory["id"])
client.export()
client.forget()
```

## API

| Route | Purpose |
|---|---|
| `POST /memories` | Add a fact with source + timestamp |
| `GET /memories/search?q=&tenant_id=&k=` | Hybrid keyword/vector retrieval with evidence |
| `GET /memories/{id}` | Fetch one memory |
| `PATCH /memories/{id}` | Correct a fact (supersedes; history preserved) |
| `DELETE /memories/{id}` | Delete a fact, including from the vector index |
| `GET /memories/{id}/history` | All versions of a fact, newest first |
| `POST /entities`, `GET /entities` | People, projects, organizations |
| `POST /relations`, `GET /entities/{id}/relations` | Relations between entities |
| `GET /export?tenant_id=` | Full export of a tenant's data |
| `POST /tenants/{id}/forget` | Delete everything for a tenant |
| `POST /admin/purge-expired` | Expire memories past `valid_to` / retention |

## Memory record

Each memory carries: `id`, `tenant_id`, `subject_id`, `memory_type` (preference / fact / event / relationship / instruction), `content`, `key` (normalized attribute for contradiction detection), `source_type` / `source_id` / `source_excerpt` provenance, `valid_from` / `valid_to` / `recorded_at` / `superseded_by`, `status` (active / superseded / expired / deleted), `contradicted`, `confidence`, `extractor_version`, `origin` (`user_stated` / `inferred`), `sensitivity_class`, `consent_basis`, `retention_days`.

## Retrieval

Hybrid search, re-ranked:

1. **Keyword** — IDF-weighted term overlap.
2. **Vector** — cosine similarity over TF-IDF vectors built from scratch (stopwords removed; query terms absent from every document get maximum IDF, so a query sharing only a name with a document scores near zero).
3. **Re-rank** — relevance 0.55, recency 0.25 (180-day half-life), source reliability 0.20. Memories with zero term overlap are never returned: recency and reliability alone are not evidence.

Contradictions: a newer fact with the same key supersedes the older one and marks it contradicted (both preserved, both queryable via history). With `on_conflict="keep_both"`, both stay active and flagged, and `answer()` returns "I don't know" for that key.

## Evaluation

`omnimemory/evaluation.py` provides: recall@K, precision@K, citation/provenance accuracy, stale-memory rate, contradiction resolution, deletion completeness, and cross-tenant isolation checks, plus an `evaluate()` harness over fixture cases. The demo prints a full evaluation report.

## Honest status

This is a **reference implementation in Python 3, standard library only**. What genuinely works: the full CRUD + search + correct + delete API (store and HTTP), temporal supersede with history, contradiction preservation and live-conflict "I don't know", hybrid TF-IDF retrieval with evidence, tenant isolation, retention/expiry, export, verified forget, entities/relations, and the evaluation suite — all covered by 18 passing tests.

What is simplified (see `ARCHITECTURE.md` for the production path):

- **In-memory store**, no persistence. Production path: PostgreSQL + pgvector.
- **TF-IDF vectors, not embeddings.** Production path: a versioned embedding provider behind the same retrieval interface.
- **No model-backed extraction.** Facts are added explicitly; the extractor worker (Redis queue + model) is a later milestone.
- **No auth, encryption, or multi-process deployment.** The HTTP server is a reference wrapper; tenant isolation is enforced at the API/store layer, not by a database.
- **Relevance floor (0.30) and re-rank weights are hand-tuned** on the demo corpus, not learned.

## Repository structure

```text
omnimemory/           # package: schema, store, retrieval, evaluation, server, client
tests/                # 18 unittest tests (store + HTTP end to end)
demo.py               # scripted lifecycle demo
ARCHITECTURE.md       # design, data model, production path
```

## License

MIT — see `LICENSE`.
