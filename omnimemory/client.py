"""Minimal Python SDK client for OMNIMEMORY (stdlib urllib only).

Example:
    client = OmniMemoryClient("http://127.0.0.1:8080", tenant_id="acme")
    memory = client.add_memory(subject_id="user-1", memory_type="preference",
                               content="Prefers dark mode",
                               source_type="user", source_id="chat:42")
    results = client.search("dark mode preferences")
    client.correct_memory(memory["id"], "Prefers light mode")
    client.delete_memory(memory["id"])
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


class OmniMemoryError(Exception):
    def __init__(self, status: int, payload: dict):
        self.status = status
        self.payload = payload
        super().__init__(f"HTTP {status}: {payload.get('message', payload)}")


class OmniMemoryClient:
    def __init__(self, base_url: str, tenant_id: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.tenant_id = tenant_id
        self.timeout = timeout

    # ------------------------------------------------------------ plumbing
    def _request(self, method: str, path: str, body: dict | None = None):
        url = self.base_url + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8") or "{}")
            raise OmniMemoryError(exc.code, payload) from exc

    # ------------------------------------------------------------ memories
    def add_memory(self, **fields) -> dict:
        body = {"tenant_id": self.tenant_id, **fields}
        return self._request("POST", "/memories", body)

    def search(self, query: str, k: int = 5, **filters) -> dict:
        params = {"q": query, "tenant_id": self.tenant_id, "k": str(k)}
        params.update({key: str(value) for key, value in filters.items()})
        return self._request("GET", "/memories/search?" + urllib.parse.urlencode(params))

    def get_memory(self, memory_id: str) -> dict:
        return self._request(
            "GET",
            f"/memories/{memory_id}?tenant_id={urllib.parse.quote(self.tenant_id)}",
        )

    def correct_memory(self, memory_id: str, content: str, **fields) -> dict:
        body = {"tenant_id": self.tenant_id, "content": content, **fields}
        return self._request("PATCH", f"/memories/{memory_id}", body)

    def delete_memory(self, memory_id: str) -> dict:
        return self._request(
            "DELETE",
            f"/memories/{memory_id}?tenant_id={urllib.parse.quote(self.tenant_id)}",
        )

    def memory_history(self, memory_id: str) -> dict:
        return self._request(
            "GET",
            f"/memories/{memory_id}/history?tenant_id={urllib.parse.quote(self.tenant_id)}",
        )

    # ------------------------------------------------------------ entities
    def add_entity(self, entity_type: str, name: str, attributes: dict | None = None) -> dict:
        return self._request(
            "POST",
            "/entities",
            {
                "tenant_id": self.tenant_id,
                "entity_type": entity_type,
                "name": name,
                "attributes": attributes or {},
            },
        )

    def list_entities(self, entity_type: str | None = None) -> dict:
        params = {"tenant_id": self.tenant_id}
        if entity_type:
            params["type"] = entity_type
        return self._request("GET", "/entities?" + urllib.parse.urlencode(params))

    # -------------------------------------------------------------- control
    def export(self) -> dict:
        return self._request(
            "GET", f"/export?tenant_id={urllib.parse.quote(self.tenant_id)}"
        )

    def forget(self) -> dict:
        return self._request(
            "POST", f"/tenants/{urllib.parse.quote(self.tenant_id)}/forget"
        )

    def health(self) -> dict:
        return self._request("GET", "/health")
