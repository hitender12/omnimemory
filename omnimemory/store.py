"""The OMNIMEMORY fact store.

Design rules enforced here:

* Corrections supersede, never silently overwrite. The old version stays in
  history with ``superseded_by`` pointing at the new version.
* Contradictions are preserved, not resolved by deletion. A newer fact with
  the same key supersedes the older one and marks it ``contradicted``; with
  ``on_conflict="keep_both"`` both stay active and are flagged as a live
  conflict, which makes :meth:`MemoryStore.answer` say "I don't know".
* Inferred facts are stored with ``origin="inferred"`` and are excluded from
  default retrieval: an inference is never presented as a confirmed user fact.
* Tenants are fully isolated: every operation is scoped to one tenant, and a
  cross-tenant access raises :class:`TenantMismatch`.
* Deletion propagates to the vector index immediately, and ``forget`` removes
  every trace of a tenant (facts, history, entities, relations, index).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .errors import MemoryNotFound, TenantMismatch, ValidationError
from .retrieval import ScoredMemory, TfidfIndex, rerank
from .schema import Entity, Memory, Relation, new_id, utcnow

_UNKNOWN = "I don't know"


class MemoryStore:
    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        self._index = TfidfIndex()
        self._seq = 0

    # ------------------------------------------------------------------ add
    def add(
        self,
        tenant_id: str,
        subject_id: str,
        memory_type: str,
        content: str,
        source_type: str,
        source_id: str,
        source_excerpt: str = "",
        key: str | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        confidence: float = 1.0,
        extractor_version: str = "manual/1.0",
        origin: str = "user_stated",
        sensitivity_class: str = "low",
        consent_basis: str = "explicit",
        retention_days: int | None = None,
        on_conflict: str = "supersede",
    ) -> Memory:
        if on_conflict not in ("supersede", "keep_both"):
            raise ValidationError("on_conflict must be 'supersede' or 'keep_both'")
        if not tenant_id or not subject_id:
            raise ValidationError("tenant_id and subject_id are required")
        memory = Memory(
            id=new_id(),
            tenant_id=tenant_id,
            subject_id=subject_id,
            memory_type=memory_type,
            content=content,
            source_type=source_type,
            source_id=source_id,
            source_excerpt=source_excerpt,
            key=key,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
            extractor_version=extractor_version,
            origin=origin,
            sensitivity_class=sensitivity_class,
            consent_basis=consent_basis,
            retention_days=retention_days,
        )
        # Contradiction handling for the same key within a tenant+subject.
        if key is not None:
            for other in self._active_for(tenant_id, subject_id):
                if (
                    other.key == key
                    and other.id != memory.id
                    and other.normalized != memory.normalized
                ):
                    if on_conflict == "supersede":
                        other.status = "superseded"
                        other.superseded_by = memory.id
                        other.contradicted = True
                        # Stays in the index: superseded versions remain
                        # retrievable history (include_history=True).
                    else:
                        other.contradicted = True
                        memory.contradicted = True
        self._memories[memory.id] = memory
        self._index.add(memory.id, _index_text(memory))
        self._seq += 1
        return memory

    # ------------------------------------------------------------------ get
    def get(self, memory_id: str, tenant_id: str) -> Memory:
        memory = self._memories.get(memory_id)
        if memory is None or memory.status == "deleted":
            raise MemoryNotFound(f"no memory {memory_id!r}")
        if memory.tenant_id != tenant_id:
            raise TenantMismatch("memory belongs to a different tenant")
        return memory

    # -------------------------------------------------------------- search
    def search(
        self,
        query: str,
        tenant_id: str,
        k: int = 5,
        memory_type: str | None = None,
        subject_id: str | None = None,
        include_history: bool = False,
        include_inferred: bool = False,
        now: datetime | None = None,
    ) -> list[ScoredMemory]:
        now = now or utcnow()
        candidates = [
            m
            for m in self._memories.values()
            if m.tenant_id == tenant_id
            and m.status in ("active", "superseded")
            and (include_history or m.status == "active")
            and (memory_type is None or m.memory_type == memory_type)
            and (subject_id is None or m.subject_id == subject_id)
            and (include_inferred or m.origin != "inferred")
            and not self._is_expired(m, now)
        ]
        if not candidates:
            return []
        ids = {m.id for m in candidates}
        scores = self._index.search(query, ids)
        scored: list[ScoredMemory] = []
        for memory in candidates:
            entry = scores.get(memory.id, {"keyword_score": 0.0, "vector_score": 0.0})
            # A memory with no term overlap at all is not evidence for the
            # query: recency and source reliability alone never justify
            # returning it.
            if entry["keyword_score"] == 0.0 and entry["vector_score"] == 0.0:
                continue
            score = rerank(memory, entry["keyword_score"], entry["vector_score"], now)
            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=score,
                    keyword_score=entry["keyword_score"],
                    vector_score=entry["vector_score"],
                    recency_score=0.0,
                    superseded_versions=self._history_of(memory),
                )
            )
        scored.sort(key=lambda s: s.score, reverse=True)
        # Mark live contradictions: same key, more than one active record.
        active_by_key: dict[str, list[ScoredMemory]] = {}
        for item in scored:
            if item.memory.status == "active" and item.memory.key is not None:
                active_by_key.setdefault(item.memory.key, []).append(item)
        for group in active_by_key.values():
            if len(group) > 1 and len({g.memory.normalized for g in group}) > 1:
                for item in group:
                    item.conflict = True
        return scored[:k]

    def answer(
        self, query: str, tenant_id: str, k: int = 5, threshold: float = 0.15
    ) -> dict:
        """Evidence-backed answer, or an honest "I don't know".

        Returns ``unknown`` when no memory matches above threshold, when
        the best match has too little *relevance* (recency and source
        reliability alone never justify an answer), or when the
        best-matching memories contradict each other.
        """
        results = self.search(query, tenant_id, k=k)
        if not results or results[0].score < threshold:
            return {"status": "unknown", "answer": _UNKNOWN, "evidence": []}
        top = results[0]
        relevance = 0.5 * top.keyword_score + 0.5 * top.vector_score
        if relevance < 0.30:
            return {
                "status": "unknown",
                "answer": _UNKNOWN,
                "reason": "no relevant memory",
                "evidence": [],
            }
        if top.conflict:
            return {
                "status": "unknown",
                "answer": _UNKNOWN,
                "reason": "conflicting memories",
                "evidence": [_evidence(r) for r in results if r.conflict],
            }
        return {
            "status": "answered",
            "answer": top.memory.content,
            "evidence": [_evidence(r) for r in results if r.score >= threshold],
        }

    # ------------------------------------------------------------ correct
    def correct(
        self,
        memory_id: str,
        tenant_id: str,
        new_content: str,
        source_type: str | None = None,
        source_id: str | None = None,
        source_excerpt: str | None = None,
        confidence: float | None = None,
    ) -> Memory:
        """Correct a fact: the old version is superseded, never overwritten."""
        old = self.get(memory_id, tenant_id)
        if old.status != "active":
            raise ValidationError("only an active memory can be corrected")
        new = Memory(
            id=new_id(),
            tenant_id=old.tenant_id,
            subject_id=old.subject_id,
            memory_type=old.memory_type,
            content=new_content,
            source_type=source_type or old.source_type,
            source_id=source_id or old.source_id,
            source_excerpt=source_excerpt if source_excerpt is not None else old.source_excerpt,
            key=old.key,
            valid_from=old.valid_from,
            valid_to=old.valid_to,
            confidence=old.confidence if confidence is None else confidence,
            extractor_version=old.extractor_version,
            origin=old.origin,
            sensitivity_class=old.sensitivity_class,
            consent_basis=old.consent_basis,
            retention_days=old.retention_days,
        )
        old.status = "superseded"
        old.superseded_by = new.id
        # The old version stays in the index so history search can find it;
        # only delete/expire removes a document from the index.
        self._memories[new.id] = new
        self._index.add(new.id, _index_text(new))
        return new

    # ------------------------------------------------------------- delete
    def delete(self, memory_id: str, tenant_id: str) -> bool:
        """Delete a fact and remove it from the derived vector index."""
        memory = self.get(memory_id, tenant_id)
        memory.status = "deleted"
        del self._memories[memory_id]
        self._index.remove(memory_id)
        return True

    def history(self, memory_id: str, tenant_id: str) -> list[Memory]:
        """All versions of a fact, newest first (current + superseded)."""
        current = self.get(memory_id, tenant_id)
        chain = [current]
        seen = {current.id}
        node = current
        # Walk backwards through superseded_by links held on older versions.
        while True:
            prev = next(
                (
                    m
                    for m in self._memories.values()
                    if m.superseded_by == node.id and m.id not in seen
                ),
                None,
            )
            if prev is None:
                break
            chain.append(prev)
            seen.add(prev.id)
            node = prev
        return chain

    # ---------------------------------------------------- retention/export
    def purge_expired(self, now: datetime | None = None) -> int:
        """Expire memories past valid_to or their retention deadline."""
        now = now or utcnow()
        count = 0
        for memory in list(self._memories.values()):
            if memory.status != "active":
                continue
            expired = False
            if memory.valid_to is not None and memory.valid_to <= now:
                expired = True
            if memory.retention_days is not None:
                age = (now - memory.recorded_at).days
                if age >= memory.retention_days:
                    expired = True
            if expired:
                memory.status = "expired"
                self._index.remove(memory.id)
                count += 1
        return count

    def export(self, tenant_id: str) -> dict:
        """Full export of a tenant's memories (current + history)."""
        records = [
            m.to_dict()
            for m in self._memories.values()
            if m.tenant_id == tenant_id
        ]
        return {
            "tenant_id": tenant_id,
            "exported_at": utcnow().isoformat(),
            "memories": records,
            "entities": [
                e.to_dict() for e in self._entities.values() if e.tenant_id == tenant_id
            ],
            "relations": [
                r.to_dict()
                for r in self._relations.values()
                if r.tenant_id == tenant_id
            ],
        }

    def forget_tenant(self, tenant_id: str) -> int:
        """Delete everything belonging to a tenant, across store and index."""
        removed = 0
        for memory_id in [
            m.id for m in self._memories.values() if m.tenant_id == tenant_id
        ]:
            del self._memories[memory_id]
            self._index.remove(memory_id)
            removed += 1
        for entity_id in [
            e.id for e in self._entities.values() if e.tenant_id == tenant_id
        ]:
            del self._entities[entity_id]
        for relation_id in [
            r.id for r in self._relations.values() if r.tenant_id == tenant_id
        ]:
            del self._relations[relation_id]
        return removed

    # ------------------------------------------------------------ entities
    def add_entity(
        self, tenant_id: str, entity_type: str, name: str, attributes: dict | None = None
    ) -> Entity:
        entity = Entity(
            id=new_id(),
            tenant_id=tenant_id,
            entity_type=entity_type,
            name=name,
            attributes=attributes or {},
        )
        self._entities[entity.id] = entity
        return entity

    def get_entity(self, entity_id: str, tenant_id: str) -> Entity:
        entity = self._entities.get(entity_id)
        if entity is None:
            raise MemoryNotFound(f"no entity {entity_id!r}")
        if entity.tenant_id != tenant_id:
            raise TenantMismatch("entity belongs to a different tenant")
        return entity

    def list_entities(
        self, tenant_id: str, entity_type: str | None = None
    ) -> list[Entity]:
        return [
            e
            for e in self._entities.values()
            if e.tenant_id == tenant_id
            and (entity_type is None or e.entity_type == entity_type)
        ]

    def add_relation(
        self, tenant_id: str, from_entity_id: str, relation: str, to_entity_id: str
    ) -> Relation:
        self.get_entity(from_entity_id, tenant_id)
        self.get_entity(to_entity_id, tenant_id)
        rel = Relation(
            id=new_id(),
            tenant_id=tenant_id,
            from_entity_id=from_entity_id,
            relation=relation,
            to_entity_id=to_entity_id,
        )
        self._relations[rel.id] = rel
        return rel

    def entity_relations(self, entity_id: str, tenant_id: str) -> list[Relation]:
        self.get_entity(entity_id, tenant_id)
        return [
            r
            for r in self._relations.values()
            if r.tenant_id == tenant_id
            and (r.from_entity_id == entity_id or r.to_entity_id == entity_id)
        ]

    # ------------------------------------------------------------- helpers
    def _active_for(self, tenant_id: str, subject_id: str) -> list[Memory]:
        return [
            m
            for m in self._memories.values()
            if m.tenant_id == tenant_id
            and m.subject_id == subject_id
            and m.status == "active"
        ]

    def _history_of(self, memory: Memory) -> list[Memory]:
        """Superseded versions that led to this memory."""
        chain: list[Memory] = []
        node = memory
        seen = set()
        while True:
            prev = next(
                (
                    m
                    for m in self._memories.values()
                    if m.superseded_by == node.id and m.id not in seen
                ),
                None,
            )
            if prev is None:
                break
            chain.append(prev)
            seen.add(prev.id)
            node = prev
        return chain

    @staticmethod
    def _is_expired(memory: Memory, now: datetime) -> bool:
        if memory.valid_to is not None and memory.valid_to <= now:
            return True
        if memory.retention_days is not None:
            return (now - memory.recorded_at).days >= memory.retention_days
        return False

    @property
    def index_size(self) -> int:
        return len(self._index)

    def index_contains(self, memory_id: str) -> bool:
        return self._index.contains(memory_id)


def _index_text(memory: Memory) -> str:
    parts = [memory.content]
    if memory.key:
        parts.append(memory.key.replace(":", " ").replace("_", " "))
    return " ".join(parts)


def _evidence(scored: ScoredMemory) -> dict:
    memory = scored.memory
    return {
        "id": memory.id,
        "content": memory.content,
        "score": round(scored.score, 4),
        "source_type": memory.source_type,
        "source_id": memory.source_id,
        "source_excerpt": memory.source_excerpt,
        "recorded_at": memory.recorded_at.isoformat(),
        "status": memory.status,
        "origin": memory.origin,
        "conflict": scored.conflict,
    }
