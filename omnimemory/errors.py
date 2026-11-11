"""Errors raised by the OMNIMEMORY store and server."""


class MemoryError(Exception):
    """Base class for store errors."""


class MemoryNotFound(MemoryError):
    """No memory with that id exists (in this tenant)."""


class TenantMismatch(MemoryError):
    """The memory exists but belongs to a different tenant."""


class ValidationError(MemoryError):
    """The request payload failed validation."""


class ConflictError(MemoryError):
    """The operation conflicts with existing state."""
