"""Backend API. All returned GPU results must be ready, not just enqueued.

Implementations receive a contracts.Clock-compatible callable. TokenResult's
ready_ns is read from that clock after GPU completion and includes output EOS.
"""

from pathlib import Path
from typing import Protocol

from ..config import RuntimeConfig
from ..contracts import CacheHandle, RequestSpec, TokenResult


class Backend(Protocol):
    def load_session(self, snapshot_path: Path, config: RuntimeConfig) -> None:
        """Load one model instance once; record actual import/model provenance."""
        ...

    def prefill_raw(self, request: RequestSpec) -> TokenResult:
        """No quantization. Return y1 and post-RoPE FP16 cache with T=L."""
        ...

    def pack_initial(self, cache: CacheHandle, group_size: int) -> None:
        """Validate all layers, pack, GPU-complete, atomically commit; keep T."""
        ...

    def decode_one(self, cache: CacheHandle, previous_token_id: int) -> TokenResult:
        """Consume last output at position T; return next token and T+1 cache."""
        ...

    def release_request(self, cache: CacheHandle) -> None:
        """Drop all request cache references; safe for repeated cleanup calls."""
        ...
