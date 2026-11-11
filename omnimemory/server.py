"""Stdlib HTTP wrapper for OMNIMEMORY.

Exposes the same operations as :class:`MemoryStore` over HTTP with no
external dependencies (the spec's FastAPI + PostgreSQL + pgvector stack is
the production path; this server is the runnable reference implementation).

Routes (all JSON):
    GET  /health
    POST /memories                      add a fact
    GET  /memories/search?q=&tenant_id=&k=&include_history=&include_inferred=
    GET  /memories/{id}?tenant_id=
    PATCH /memories/{id}                correct a fact (supersedes)
    DELETE /memories/{id}?tenant_id=
    GET  /memories/{id}/history?tenant_id=
    POST /entities                      add person/project/organization
    GET  /entities?tenant_id=&type=
    POST /relations
    GET  /entities/{id}/relations?tenant_id=
    GET  /export?tenant_id=
    POST /tenants/{tenant_id}/forget
    POST /admin/purge-expired

Tenant isolation is enforced on every route: a missing tenant_id is a 400,
a cross-tenant access is a 403.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .errors import MemoryError, MemoryNotFound, TenantMismatch, ValidationError
from .store import MemoryStore

_ISO_FORMATS = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")


def _parse_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValidationError(f"bad datetime: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes", "on")


class MemoryHandler(BaseHTTPRequestHandler):
    store: MemoryStore = None  # set by make_server
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------- plumbing
    def log_message(self, *args):  # keep test output clean
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ValidationError("request body is not valid JSON")
        if not isinstance(data, dict):
            raise ValidationError("request body must be a JSON object")
        return data

    def _tenant(self, query: dict, body: dict | None = None) -> str:
        tenant = (query.get("tenant_id") or [None])[0]
        if tenant is None and body:
            tenant = body.get("tenant_id")
        if not tenant:
            raise ValidationError("tenant_id is required")
        return tenant

    def _handle(self, fn):
        try:
            fn()
        except MemoryNotFound as exc:
            self._send(404, {"error": "not_found", "message": str(exc)})
        except TenantMismatch as exc:
            self._send(403, {"error": "tenant_mismatch", "message": str(exc)})
        except ValidationError as exc:
            self._send(400, {"error": "validation_error", "message": str(exc)})
        except MemoryError as exc:
            self._send(400, {"error": "memory_error", "message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            self._send(500, {"error": "internal_error", "message": str(exc)})

    # -------------------------------------------------------------- routing
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"

        def run():
            if path == "/health":
                self._send(200, {"status": "ok"})
            elif path == "/memories/search":
                self._search(query)
            elif path == "/export":
                tenant = self._tenant(query)
                self._send(200, self.store.export(tenant))
            elif path == "/entities":
                tenant = self._tenant(query)
                etype = (query.get("type") or [None])[0]
                self._send(
                    200,
                    {
                        "entities": [
                            e.to_dict()
                            for e in self.store.list_entities(tenant, etype)
                        ]
                    },
                )
            elif path.startswith("/memories/") and path.endswith("/history"):
                memory_id = path.split("/")[2]
                tenant = self._tenant(query)
                self._send(
                    200,
                    {
                        "history": [
                            m.to_dict()
                            for m in self.store.history(memory_id, tenant)
                        ]
                    },
                )
            elif path.startswith("/memories/"):
                memory_id = path.split("/")[2]
                tenant = self._tenant(query)
                memory = self.store.get(memory_id, tenant)
                self._send(200, memory.to_dict())
            elif path.startswith("/entities/") and path.endswith("/relations"):
                entity_id = path.split("/")[2]
                tenant = self._tenant(query)
                self._send(
                    200,
                    {
                        "relations": [
                            r.to_dict()
                            for r in self.store.entity_relations(entity_id, tenant)
                        ]
                    },
                )
            else:
                self._send(404, {"error": "not_found", "message": "unknown route"})

        self._handle(run)

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"

        def run():
            body = self._read_json()
            if path == "/memories":
                self._add_memory(body)
            elif path == "/entities":
                tenant = self._tenant(query, body)
                entity = self.store.add_entity(
                    tenant,
                    body["entity_type"],
                    body["name"],
                    body.get("attributes") or {},
                )
                self._send(201, entity.to_dict())
            elif path == "/relations":
                tenant = self._tenant(query, body)
                rel = self.store.add_relation(
                    tenant,
                    body["from_entity_id"],
                    body["relation"],
                    body["to_entity_id"],
                )
                self._send(201, rel.to_dict())
            elif path == "/admin/purge-expired":
                count = self.store.purge_expired()
                self._send(200, {"expired": count})
            elif path.startswith("/tenants/") and path.endswith("/forget"):
                tenant_id = path.split("/")[2]
                removed = self.store.forget_tenant(tenant_id)
                self._send(200, {"tenant_id": tenant_id, "deleted_memories": removed})
            else:
                self._send(404, {"error": "not_found", "message": "unknown route"})

        self._handle(run)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"

        def run():
            if not path.startswith("/memories/"):
                self._send(404, {"error": "not_found", "message": "unknown route"})
                return
            memory_id = path.split("/")[2]
            body = self._read_json()
            tenant = self._tenant(query, body)
            if "content" not in body:
                raise ValidationError("PATCH requires 'content'")
            memory = self.store.correct(
                memory_id,
                tenant,
                body["content"],
                source_type=body.get("source_type"),
                source_id=body.get("source_id"),
                source_excerpt=body.get("source_excerpt"),
                confidence=body.get("confidence"),
            )
            self._send(200, memory.to_dict())

        self._handle(run)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"

        def run():
            if not path.startswith("/memories/"):
                self._send(404, {"error": "not_found", "message": "unknown route"})
                return
            memory_id = path.split("/")[2]
            tenant = self._tenant(query)
            self.store.delete(memory_id, tenant)
            self._send(200, {"deleted": memory_id})

        self._handle(run)

    # -------------------------------------------------------------- helpers
    def _add_memory(self, body: dict) -> None:
        tenant = self._tenant({}, body)
        for field in ("subject_id", "memory_type", "content", "source_type", "source_id"):
            if field not in body:
                raise ValidationError(f"missing required field: {field}")
        memory = self.store.add(
            tenant_id=tenant,
            subject_id=body["subject_id"],
            memory_type=body["memory_type"],
            content=body["content"],
            source_type=body["source_type"],
            source_id=body["source_id"],
            source_excerpt=body.get("source_excerpt", ""),
            key=body.get("key"),
            valid_from=_parse_datetime(body.get("valid_from")),
            valid_to=_parse_datetime(body.get("valid_to")),
            confidence=float(body.get("confidence", 1.0)),
            extractor_version=body.get("extractor_version", "manual/1.0"),
            origin=body.get("origin", "user_stated"),
            sensitivity_class=body.get("sensitivity_class", "low"),
            consent_basis=body.get("consent_basis", "explicit"),
            retention_days=body.get("retention_days"),
            on_conflict=body.get("on_conflict", "supersede"),
        )
        self._send(201, memory.to_dict())

    def _search(self, query: dict) -> None:
        tenant = self._tenant(query)
        q = (query.get("q") or [""])[0]
        k = int((query.get("k") or ["5"])[0])
        results = self.store.search(
            q,
            tenant,
            k=k,
            memory_type=(query.get("memory_type") or [None])[0],
            subject_id=(query.get("subject_id") or [None])[0],
            include_history=_parse_bool((query.get("include_history") or [None])[0]),
            include_inferred=_parse_bool((query.get("include_inferred") or [None])[0]),
        )
        self._send(
            200,
            {
                "query": q,
                "tenant_id": tenant,
                "results": [
                    {
                        "memory": r.memory.to_dict(),
                        "score": round(r.score, 4),
                        "keyword_score": round(r.keyword_score, 4),
                        "vector_score": round(r.vector_score, 4),
                        "conflict": r.conflict,
                        "superseded_versions": [
                            m.to_dict() for m in r.superseded_versions
                        ],
                    }
                    for r in results
                ],
            },
        )


def make_server(store: MemoryStore | None = None, host: str = "127.0.0.1", port: int = 0):
    """Build a ThreadingHTTPServer bound to an ephemeral port by default."""
    store = store or MemoryStore()

    class Handler(MemoryHandler):
        pass

    Handler.store = store
    server = ThreadingHTTPServer((host, port), Handler)
    return server, store


def main() -> None:
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server, _ = make_server(port=port)
    print(f"OMNIMEMORY listening on http://127.0.0.1:{port}")
    print("Routes: POST /memories, GET /memories/search, PATCH /memories/{id}, DELETE /memories/{id}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
