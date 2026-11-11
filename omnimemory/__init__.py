"""OMNIMEMORY — a user-controlled long-term memory API for AI apps.

Durable facts, temporal updates, hybrid retrieval and source-backed results.
"""

from .client import OmniMemoryClient, OmniMemoryError
from .errors import (
    ConflictError,
    MemoryError,
    MemoryNotFound,
    TenantMismatch,
    ValidationError,
)
from .evaluation import (
    citation_accuracy,
    contradiction_resolution_check,
    deletion_completeness_check,
    evaluate,
    precision_at_k,
    recall_at_k,
    stale_memory_rate,
    tenant_isolation_check,
)
from .schema import Entity, Memory, Relation
from .server import make_server
from .store import MemoryStore

__all__ = [
    "MemoryStore",
    "Memory",
    "Entity",
    "Relation",
    "OmniMemoryClient",
    "OmniMemoryError",
    "MemoryError",
    "MemoryNotFound",
    "TenantMismatch",
    "ValidationError",
    "ConflictError",
    "make_server",
    "recall_at_k",
    "precision_at_k",
    "citation_accuracy",
    "stale_memory_rate",
    "contradiction_resolution_check",
    "deletion_completeness_check",
    "tenant_isolation_check",
    "evaluate",
]

__version__ = "0.1.0"
