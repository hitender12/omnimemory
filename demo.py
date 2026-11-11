"""OMNIMEMORY demo: one user's memory lifecycle, end to end.

Run:  python demo.py

Shows: add -> hybrid search -> correction (supersede, history preserved) ->
live contradiction -> honest "I don't know" -> delete + index cleanup ->
export -> forget -> evaluation metrics -> HTTP server smoke test.
"""

from __future__ import annotations

import json
import threading
import urllib.request

from omnimemory import MemoryStore, evaluate
from omnimemory.evaluation import (
    contradiction_resolution_check,
    deletion_completeness_check,
    tenant_isolation_check,
)
from omnimemory.server import make_server

TENANT = "acme"
USER = "user-1"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    store = MemoryStore()

    section("1. Add memories (user-stated facts, with provenance)")
    name = store.add(
        TENANT, USER, "fact", "The user's name is Maya Chen.",
        source_type="user", source_id="chat:101", source_excerpt="Hi, I'm Maya",
        key="fact:name",
    )
    coffee = store.add(
        TENANT, USER, "preference", "Maya prefers oat milk lattes.",
        source_type="conversation", source_id="chat:104",
        source_excerpt="oat milk latte please", key="preference:coffee",
    )
    project = store.add(
        TENANT, USER, "fact", "Maya leads Project Northstar, a billing migration.",
        source_type="document", source_id="doc:77",
        source_excerpt="Northstar owner: Maya Chen", key="fact:project",
    )
    store.add(
        TENANT, USER, "preference", "Maya likes dark mode in every app.",
        source_type="user", source_id="chat:109", key="preference:theme",
    )
    # An inference is stored but never presented as a confirmed fact.
    store.add(
        TENANT, USER, "preference", "Maya probably likes jazz music.",
        source_type="inference", source_id="model:extract-v1",
        origin="inferred", confidence=0.4, key="preference:music",
    )
    print(f"added {len(store._memories)} memories (1 inferred, excluded by default)")

    section("2. Hybrid search returns evidence, not just text")
    for result in store.search("What coffee does Maya like?", TENANT, k=3):
        print(f"  [{result.score:.3f}] {result.memory.content}"
              f"  (source {result.memory.source_id}, {result.memory.recorded_at.date()})")

    section("3. Correction supersedes — history is preserved, never overwritten")
    fixed = store.correct(
        coffee.id, TENANT, "Maya prefers black coffee, no milk.",
        source_type="user", source_id="chat:120",
    )
    print(f"  new version: {fixed.content}")
    print("  history:")
    for version in store.history(fixed.id, TENANT):
        print(f"    - [{version.status}] {version.content}")

    section("4. Contradiction: the same key, a different value")
    store.add(
        TENANT, USER, "fact", "Maya leads Project Northstar.",
        source_type="user", source_id="chat:130", key="fact:project",
    )
    check = contradiction_resolution_check(store, "fact:project", TENANT)
    print(f"  resolution check: {check}")
    print("  both versions kept; old one marked superseded + contradicted")

    section("5. Live conflict -> the system says \"I don't know\"")
    store.add(
        TENANT, USER, "fact", "Maya's home city is Lisbon.",
        source_type="user", source_id="chat:140", key="fact:home_city",
        on_conflict="keep_both",
    )
    store.add(
        TENANT, USER, "fact", "Maya's home city is Porto.",
        source_type="document", source_id="doc:88", key="fact:home_city",
        on_conflict="keep_both",
    )
    for question in (
        "Where is Maya's home city?",
        "What project is Maya leading?",
        "What is Maya's favourite film?",
    ):
        print("  answer:", store.answer(question, TENANT))

    section("6. Delete removes the fact from search AND the vector index")
    store.delete(project.id, TENANT)
    print(f"  deleted {project.id}; index contains it: {store.index_contains(project.id)}")
    print("  " + str(deletion_completeness_check(store, TENANT, project.id, "Northstar")))

    section("7. Entities and relations")
    maya = store.add_entity(TENANT, "person", "Maya Chen", {"role": "engineer"})
    northstar = store.add_entity(TENANT, "project", "Project Northstar", {"status": "active"})
    rel = store.add_relation(TENANT, maya.id, "leads", northstar.id)
    print(f"  {maya.name} --{rel.relation}--> {northstar.name}")

    section("8. Tenant isolation")
    store.add(
        "other-corp", "user-9", "fact", "The vault password is hunter2.",
        source_type="user", source_id="chat:1", key="fact:vault",
    )
    print("  " + str(tenant_isolation_check(store, "other-corp", TENANT, "vault password")))

    section("9. Evaluation harness")
    cases = [
        {
            "query": "What coffee does Maya like?",
            "tenant_id": TENANT,
            "relevant_ids": [fixed.id],
            "expected_sources": {"chat:120"},
        },
        {
            "query": "Maya project Northstar",
            "tenant_id": TENANT,
            "relevant_ids": [store.search("Northstar", TENANT, k=1)[0].memory.id],
            "expected_sources": {"chat:130"},
        },
        {
            "query": "dark mode preference",
            "tenant_id": TENANT,
            "relevant_ids": [
                m.id for m in store._memories.values()
                if "dark mode" in m.content
            ],
            "expected_sources": {"chat:109"},
        },
    ]
    print("  " + json.dumps(evaluate(store, cases, k=3), indent=2))

    section("10. Export, then forget the whole tenant")
    exported = store.export(TENANT)
    print(f"  exported {len(exported['memories'])} memories, "
          f"{len(exported['entities'])} entities")
    removed = store.forget_tenant(TENANT)
    print(f"  forgot tenant {TENANT!r}: {removed} memories removed, "
          f"search now returns {store.search('Maya', TENANT)}")

    section("11. HTTP server smoke test (stdlib only)")
    server, http_store = make_server(port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/health") as resp:
            print("  health:", json.loads(resp.read()))
        payload = json.dumps({
            "tenant_id": "demo", "subject_id": "user-1", "memory_type": "fact",
            "content": "Demo user likes tea.", "source_type": "user",
            "source_id": "chat:1", "key": "fact:drink",
        }).encode()
        req = urllib.request.Request(
            base + "/memories", data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            created = json.loads(resp.read())
        print("  POST /memories ->", created["id"], created["content"])
        with urllib.request.urlopen(base + "/memories/search?q=tea&tenant_id=demo") as resp:
            found = json.loads(resp.read())
        print("  GET /memories/search ->", len(found["results"]), "result(s)")
    finally:
        server.shutdown()
        server.server_close()

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
