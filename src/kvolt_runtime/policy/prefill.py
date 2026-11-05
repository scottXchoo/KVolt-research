"""CPU-only Prefill energy selection inside the online PC phase."""
import time

from ..config import require_int
from ..errors import ContractError
from .types import (
    Bracket, FREQUENCIES_MHZ, LUTRepository, PrefillCandidate, PrefillDecision,
    PrefillPrediction, elapsed_since, finite, length as validate_length,
)


def _validate_prefill(predictions, requested_length):
    if type(predictions) is not tuple or len(predictions) != len(FREQUENCIES_MHZ):
        raise ContractError('Prefill LUT must return an immutable tuple of all eight frequencies')
    normalized = []
    for row in predictions:
        if not isinstance(row, PrefillPrediction):
            raise ContractError('Prefill LUT must return PrefillPrediction rows')
        if not isinstance(row.bracket, Bracket):
            raise ContractError('Prefill prediction requires an interpolation bracket')
        # Revalidate even frozen dataclasses supplied by a foreign repository.
        bracket = Bracket(row.bracket.lower, row.bracket.upper, row.bracket.weight)
        prediction = PrefillPrediction(row.length, row.frequency_mhz, row.ttft_s, row.energy_j, bracket)
        if row.length != requested_length:
            raise ContractError('Prefill LUT rows must match requested L')
        normalized.append(prediction)
    if {row.frequency_mhz for row in predictions} != set(FREQUENCIES_MHZ):
        raise ContractError('Prefill LUT must contain every original frequency exactly once')
    return tuple(sorted(normalized, key=lambda row: row.frequency_mhz))


class PrefillController:
    """Query all Prefill rows and their target, then read the injected tp clock.

    The M3 caller performs full-support preflight before entering this phase.
    This controller does no device/file work or request-result caching.
    """

    def __init__(self, repo: LUTRepository, *, clock=time.monotonic_ns):
        self.repo = repo
        self.clock = clock

    def choose(self, length: int, t0_ns: int) -> PrefillDecision:
        validate_length(length)
        require_int(t0_ns, 'Prefill start ns', 0)

        predictions = _validate_prefill(self.repo.prefill(length), length)
        target = finite(self.repo.prefill_target(length), 'Prefill target seconds', positive=True)
        tp_ns = self.clock()
        elapsed_ns = elapsed_since(t0_ns, tp_ns)
        budget = target - elapsed_ns / 1_000_000_000

        feasible = tuple(row for row in predictions if row.ttft_s <= budget)
        if feasible:
            selected = min(feasible, key=lambda row: (row.energy_j, row.ttft_s, row.frequency_mhz))
            selection_path = 'energy_min'
        else:
            selected = min(predictions, key=lambda row: (row.ttft_s, row.energy_j, row.frequency_mhz))
            selection_path = 'latency_fallback'

        candidates = []
        for row in predictions:
            meets_budget = row.ttft_s <= budget
            is_selected = row is selected
            reasons = []
            if not meets_budget:
                reasons.append('time_budget_exceeded')
            if not is_selected:
                reasons.append('not_selected')
            elif selection_path == 'latency_fallback':
                reasons.append('latency_fallback')
            candidates.append(PrefillCandidate(row, meets_budget, is_selected, tuple(reasons)))

        return PrefillDecision(
            length=length, selected=selected, target_s=target, budget_s=budget,
            elapsed_ns=elapsed_ns, selection_path=selection_path,
            predicted_feasible=selected.ttft_s <= budget, candidates=tuple(candidates),
            t0_ns=t0_ns, tp_ns=tp_ns)
