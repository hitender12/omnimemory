"""Tests for OMNIMEMORY.

Run:  python -m unittest discover -s tests -v
"""

import json
import threading
import unittest
import urllib.request
from datetime import datetime, timedelta, timezone

from omnimemory import MemoryStore
from omnimemory.client import OmniMemoryClient
from omnimemory.evaluation import (
    citation_accuracy,
    contradiction_resolution_check,
    deletion_completeness_check,
    evaluate,
    precision_at_k,
    recall_at_k,
    stale_memory_rate,
    tenant_isolation_check,
)
from omnimemory.server import make_server


def _add(store, tenant="acme", subject="user-1", **kw):
    fields = {
        "tenant_id": tenant,
        "subject_id": subject,
        "memory_type": "fact",
        "content": "Maya likes tea.",
        "source_type": "user",
        "source_id": "chat:1",
    }
    fields.update(kw)
    return store.add(**fields)


class TestRoundTrip(unittest.TestCase):
    def test_add_search_correct_delete(self):
        store = MemoryStore()
        memory = _add(store, key="fact:drink", content="Maya likes tea.")
        found = store.search("tea", "acme")
        self.assertEqual([r.memory.id for r in found], [memory.id])
        fixed = store.correct(memory.id, "acme", "Maya likes coffee.")
        self.assertEqual(fixed.content, "Maya likes coffee.")
        self.assertEqual(store.search("coffee", "acme")[0].memory.id, fixed.id)
        self.assertEqual(store.search("tea", "acme"), [])
        self.assertTrue(store.delete(fixed.id, "acme"))
        self.assertEqual(store.search("coffee", "acme"), [])

    def test_supersede_preserves_history(self):
        store = MemoryStore()
        original = _add(store, key="fact:drink", content="Maya likes tea.")
        fixed = store.correct(original.id, "acme", "Maya likes coffee.")
        history = store.history(fixed.id, "acme")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].status, "active")
        self.assertEqual(history[1].status, "superseded")
        self.assertEqual(history[1].superseded_by, fixed.id)
        self.assertEqual(history[1].content, "Maya likes tea.")
        # History is visible through search with include_history.
        results = store.search("tea", "acme", include_history=True)
        self.assertTrue(any(r.memory.status == "superseded" for r in results))

    def test_contradiction_returns_both_versions_marked(self):
        store = MemoryStore()
        first = _add(store, key="fact:city", content="Maya lives in Lisbon.")
        store.add(
            "acme", "user-1", "fact", "Maya lives in Porto.",
            source_type="user", source_id="chat:2", key="fact:city",
        )
        check = contradiction_resolution_check(store, "fact:city", "acme")
        self.assertTrue(check["passed"], check)
        results = store.search("Lisbon Porto city", "acme", include_history=True)
        by_id = {r.memory.id: r for r in results}
        self.assertEqual(by_id[first.id].memory.status, "superseded")
        self.assertTrue(by_id[first.id].memory.contradicted)

    def test_live_contradiction_says_i_dont_know(self):
        store = MemoryStore()
        store.add(
            "acme", "user-1", "fact", "Maya's home city is Lisbon.",
            source_type="user", source_id="chat:1", key="fact:home_city",
            on_conflict="keep_both",
        )
        store.add(
            "acme", "user-1", "fact", "Maya's home city is Porto.",
            source_type="document", source_id="doc:1", key="fact:home_city",
            on_conflict="keep_both",
        )
        answer = store.answer("Where is Maya's home city?", "acme")
        self.assertEqual(answer["status"], "unknown")
        self.assertEqual(answer["answer"], "I don't know")
        self.assertEqual(answer["reason"], "conflicting memories")

    def test_delete_removes_from_search_and_index(self):
        store = MemoryStore()
        memory = _add(store, key="fact:drink", content="Maya likes tea.")
        self.assertTrue(store.index_contains(memory.id))
        store.delete(memory.id, "acme")
        self.assertFalse(store.index_contains(memory.id))
        check = deletion_completeness_check(store, "acme", memory.id, "tea")
        self.assertTrue(check["passed"], check)

    def test_tenant_isolation(self):
        store = MemoryStore()
        _add(store, tenant="acme", content="Acme secret plan.")
        _add(store, tenant="other", content="Other secret plan.")
        self.assertEqual(store.search("secret plan", "acme")[0].memory.tenant_id, "acme")
        check = tenant_isolation_check(store, "other", "acme", "secret plan")
        self.assertTrue(check["passed"], check)
        # Cross-tenant read is a hard error, not an empty result.
        other = _add(store, tenant="other", content="Other private fact.")
        with self.assertRaises(Exception):
            store.get(other.id, "acme")

    def test_i_dont_know_on_missing_memory(self):
        store = MemoryStore()
        _add(store, content="Maya likes tea.")
        answer = store.answer("What is Maya's favourite film?", "acme")
        self.assertEqual(answer["status"], "unknown")
        self.assertEqual(answer["answer"], "I don't know")
        # Empty store also says so.
        self.assertEqual(
            MemoryStore().answer("anything at all", "acme")["answer"], "I don't know"
        )

    def test_inferred_facts_are_not_presented_as_user_facts(self):
        store = MemoryStore()
        store.add(
            "acme", "user-1", "preference", "Maya probably likes jazz.",
            source_type="inference", source_id="model:v1", origin="inferred",
            confidence=0.4,
        )
        self.assertEqual(store.search("jazz", "acme"), [])
        results = store.search("jazz", "acme", include_inferred=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory.origin, "inferred")

    def test_retention_and_expiry(self):
        store = MemoryStore()
        past = datetime.now(timezone.utc) - timedelta(days=1)
        memory = _add(store, valid_to=past, content="Temporary note about tea.")
        self.assertEqual(store.search("tea", "acme"), [])
        self.assertEqual(store.purge_expired(), 1)
        self.assertEqual(store.get(memory.id, "acme").status, "expired")
        short = _add(
            store, retention_days=0, content="Expires immediately.",
        )
        self.assertEqual(store.purge_expired(), 1)
        self.assertEqual(store.get(short.id, "acme").status, "expired")

    def test_export_and_forget(self):
        store = MemoryStore()
        _add(store, content="Maya likes tea.")
        entity = store.add_entity("acme", "person", "Maya Chen")
        exported = store.export("acme")
        self.assertEqual(len(exported["memories"]), 1)
        self.assertEqual(exported["entities"][0]["id"], entity.id)
        removed = store.forget_tenant("acme")
        self.assertEqual(removed, 1)
        self.assertEqual(store.search("tea", "acme"), [])
        self.assertEqual(store.export("acme")["memories"], [])
        self.assertEqual(store.list_entities("acme"), [])

    def test_entities_and_relations(self):
        store = MemoryStore()
        maya = store.add_entity("acme", "person", "Maya Chen", {"role": "engineer"})
        project = store.add_entity("acme", "project", "Northstar")
        rel = store.add_relation("acme", maya.id, "leads", project.id)
        self.assertEqual(rel.relation, "leads")
        relations = store.entity_relations(maya.id, "acme")
        self.assertEqual(len(relations), 1)
        self.assertEqual(store.list_entities("acme", "project")[0].id, project.id)
        with self.assertRaises(Exception):
            store.add_relation("acme", maya.id, "leads", "missing-id")


class TestEvaluation(unittest.TestCase):
    def test_metrics(self):
        self.assertAlmostEqual(recall_at_k(["a", "b"], {"a", "c"}, 2), 0.5)
        self.assertAlmostEqual(precision_at_k(["a", "b"], {"a", "c"}, 2), 0.5)
        self.assertAlmostEqual(recall_at_k([], {"a"}, 5), 0.0)

    def test_evaluate_harness(self):
        store = MemoryStore()
        tea = _add(store, content="Maya likes tea.", source_id="chat:1")
        _add(store, content="Maya likes coffee.", source_id="chat:2")
        report = evaluate(
            store,
            [
                {
                    "query": "tea",
                    "tenant_id": "acme",
                    "relevant_ids": [tea.id],
                    "expected_sources": {"chat:1"},
                }
            ],
            k=3,
        )
        self.assertEqual(report["mean_recall@k"], 1.0)
        self.assertEqual(report["mean_citation_accuracy"], 1.0)

    def test_citation_and_stale_rates(self):
        store = MemoryStore()
        memory = _add(store, content="Maya likes tea.", source_id="chat:1")
        results = store.search("tea", "acme")
        self.assertEqual(citation_accuracy(results, {"chat:1"}), 1.0)
        self.assertEqual(citation_accuracy(results, {"chat:9"}), 0.0)
        self.assertEqual(stale_memory_rate(results), 0.0)


class TestHttpServer(unittest.TestCase):
    def setUp(self):
        self.server, self.store = make_server(port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = OmniMemoryClient(f"http://127.0.0.1:{self.port}", "acme")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_http_round_trip(self):
        self.assertEqual(self.client.health(), {"status": "ok"})
        created = self.client.add_memory(
            subject_id="user-1", memory_type="fact", content="Maya likes tea.",
            source_type="user", source_id="chat:1", key="fact:drink",
        )
        self.assertEqual(created["status"], "active")
        found = self.client.search("tea")
        self.assertEqual(len(found["results"]), 1)
        self.assertEqual(found["results"][0]["memory"]["id"], created["id"])
        fixed = self.client.correct_memory(created["id"], "Maya likes coffee.")
        self.assertEqual(fixed["content"], "Maya likes coffee.")
        history = self.client.memory_history(fixed["id"])
        self.assertEqual(len(history["history"]), 2)
        self.assertEqual(self.client.delete_memory(fixed["id"]), {"deleted": fixed["id"]})
        self.assertEqual(self.client.search("coffee")["results"], [])

    def test_http_tenant_isolation(self):
        self.client.add_memory(
            subject_id="user-1", memory_type="fact", content="Acme secret.",
            source_type="user", source_id="chat:1",
        )
        other = OmniMemoryClient(f"http://127.0.0.1:{self.port}", "other")
        self.assertEqual(other.search("secret")["results"], [])

    def test_http_validation_and_404(self):
        with self.assertRaises(Exception):
            self.client.get_memory("no-such-id")
        with self.assertRaises(Exception):
            self.client.add_memory(subject_id="user-1", memory_type="fact")

    def test_http_forget(self):
        self.client.add_memory(
            subject_id="user-1", memory_type="fact", content="Maya likes tea.",
            source_type="user", source_id="chat:1",
        )
        result = self.client.forget()
        self.assertEqual(result["deleted_memories"], 1)
        self.assertEqual(self.client.search("tea")["results"], [])


if __name__ == "__main__":
    unittest.main()
