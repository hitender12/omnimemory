"""Memory record schema for OMNIMEMORY.

Every memory is a durable, inspectable fact with full provenance, temporal
metadata and user-control fields. The schema deliberately distinguishes
user-stated facts from inferred facts: an inference is never presented as a
confirmed user fact.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

MEMORY_TYPES = (
    "preference",
    "fact",
    "event",
    "relationship",
    "instruction",
)

SENSITIVITY_CLASSES = ("low", "medium", "high", "restricted")

ORIGINS = ("user_stated", "inferred")

STATUSES = ("active", "superseded", "expired", "deleted")

SOURCE_RELIABILITY = {
    "user": 1.0,
    "conversation": 0.95,
    "document": 0.85,
    "application": 0.8,
    "inference": 0.5,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def normalize_content(content: str) -> str:
    """Normalized form used for contradiction detection and dedup."""
    return re.sub(r"\s+", " ", content.strip().lower())


@dataclass
class Memory:
    id: str
    tenant_id: str
    subject_id: str
    memory_type: str
    content: str
    source_type: str
    source_id: str
    source_excerpt: str = ""
    key: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    recorded_at: datetime = field(default_factory=utcnow)
    superseded_by: str | None = None
    status: str = "active"
    contradicted: bool = False
    confidence: float = 1.0
    extractor_version: str = "manual/1.0"
    origin: str = "user_stated"
    sensitivity_class: str = "low"
    consent_basis: str = "explicit"
    retention_days: int | None = None

    def __post_init__(self) -> None:
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(f"unknown memory_type: {self.memory_type!r}")
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown origin: {self.origin!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status: {self.status!r}")
        if self.sensitivity_class not in SENSITIVITY_CLASSES:
            raise ValueError(
                f"unknown sensitivity_class: {self.sensitivity_class!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.content.strip():
            raise ValueError("content must not be empty")
        if not self.tenant_id or not self.subject_id:
            raise ValueError("tenant_id and subject_id are required")

    @property
    def normalized(self) -> str:
        return normalize_content(self.content)

    @property
    def is_current(self) -> bool:
        return self.status == "active"

    def to_dict(self, include_history: bool = False) -> dict:
        data = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "subject_id": self.subject_id,
            "memory_type": self.memory_type,
            "key": self.key,
            "content": self.content,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_excerpt": self.source_excerpt,
            "valid_from": _iso(self.valid_from),
            "valid_to": _iso(self.valid_to),
            "recorded_at": _iso(self.recorded_at),
            "superseded_by": self.superseded_by,
            "status": self.status,
            "contradicted": self.contradicted,
            "confidence": self.confidence,
            "extractor_version": self.extractor_version,
            "origin": self.origin,
            "sensitivity_class": self.sensitivity_class,
            "consent_basis": self.consent_basis,
            "retention_days": self.retention_days,
        }
        return data


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass
class Entity:
    id: str
    tenant_id: str
    entity_type: str  # person | project | organization
    name: str
    attributes: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if self.entity_type not in ("person", "project", "organization"):
            raise ValueError(f"unknown entity_type: {self.entity_type!r}")
        if not self.name.strip():
            raise ValueError("entity name must not be empty")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "attributes": self.attributes,
            "created_at": _iso(self.created_at),
        }


@dataclass
class Relation:
    id: str
    tenant_id: str
    from_entity_id: str
    relation: str
    to_entity_id: str
    created_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "from_entity_id": self.from_entity_id,
            "relation": self.relation,
            "to_entity_id": self.to_entity_id,
            "created_at": _iso(self.created_at),
        }
