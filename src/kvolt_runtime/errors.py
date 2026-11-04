"""Stable error categories; hardware/backend errors preserve their cause."""


class KVoltError(Exception):
    """Base runtime error."""


class ContractError(KVoltError, ValueError):
    """Invalid input or metadata; reject before executing GPU work."""


class CacheStateError(KVoltError, RuntimeError):
    """Invalid cache transition or use after release."""


class BackendError(KVoltError, RuntimeError):
    """Technical execution failure; caller must release the request."""


class QualificationError(BackendError):
    """A required exact comparison or provenance check failed."""
