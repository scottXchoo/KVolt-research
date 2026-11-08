"""CPU-only Prefill energy selection inside the online PC phase."""
import time

from ..config import require_int
from ..errors import ContractError
from .types import (
    Bracket, LUTRepository, PrefillCandidate, PrefillDecision, PrefillBudgetLookup,
    PrefillPrediction, elapsed_since, finite, length as validate_length,
)


def _validate_lookup(lookup, requested_length, budget):
    if not isinstance(lookup, PrefillBudgetLookup):
        raise ContractError('Prefill LUT must return PrefillBudgetLookup')
    if not isinstance(lookup.bracket, Bracket):
        raise ContractError('Prefill lookup requires an interpolation bracket')
    bracket = Bracket(lookup.bracket.lower, lookup.bracket.upper, lookup.bracket.weight)
    # Revalidate even frozen dataclasses supplied by a foreign repository.
    lookup = PrefillBudgetLookup(lookup.length, lookup.predictions, lookup.budget_s, bracket)
    if lookup.length != requested_length or lookup.budget_s != budget:
        raise ContractError('Prefill lookup must match requested L and budget')
    if not lookup.predictions:
        raise ContractError('Prefill filtered lookup must contain candidates or fastest ties')
    normalized = []
    for row in lookup.predictions:
        if not isinstance(row, PrefillPrediction):
            raise ContractError('Prefill filtered LUT must return PrefillPrediction rows')
        if not isinstance(row.bracket, Bracket):
            raise ContractError('Prefill prediction requires an interpolation bracket')
        row_bracket = Bracket(row.bracket.lower, row.bracket.upper, row.bracket.weight)
        prediction = PrefillPrediction(row.length, row.frequency_mhz, row.ttft_s, row.energy_j, row_bracket)
        if prediction.length != requested_length or prediction.bracket != bracket:
            raise ContractError('Prefill rows must match requested L and lookup interpolation bracket')
        normalized.append(prediction)
    if len({row.frequency_mhz for row in normalized}) != len(normalized):
        raise ContractError('Prefill filtered LUT must contain unique frequencies')
    feasible = [row for row in normalized if row.ttft_s <= budget]
    if feasible and len(feasible) != len(normalized):
        raise ContractError('Prefill filtered LUT mixed passing and rejected candidates')
    if not feasible and len({row.ttft_s for row in normalized}) != 1:
        raise ContractError('Prefill fallback LUT must contain only minimum-TTFT exact ties')
    return tuple(sorted(normalized, key=lambda row: row.frequency_mhz))


class PrefillController:
    """Fix the budget immediately after Reference lookup, then query candidates.

    The M3 caller performs full-support preflight before entering this phase.
    This controller does no device/file work or request-result caching.
    """

    def __init__(self, repo: LUTRepository, *, clock=time.monotonic_ns):
        self.repo = repo
        self.clock = clock

    def choose(self, length: int, t0_ns: int) -> PrefillDecision:
        validate_length(length)
        require_int(t0_ns, 'Prefill start ns', 0)
        target = finite(self.repo.prefill_target(length), 'Prefill target seconds', positive=True)
        tp_ns = self.clock()
        elapsed_ns = elapsed_since(t0_ns, tp_ns)
        budget = target - elapsed_ns / 1_000_000_000
        predictions = _validate_lookup(self.repo.prefill_within_budget(length, budget), length, budget)
        lookup_end_ns = self.clock()
        elapsed_since(tp_ns, lookup_end_ns)

        feasible = tuple(row for row in predictions if row.ttft_s <= budget)
        if feasible:
            selected = min(feasible, key=lambda row: (row.energy_j, row.ttft_s, row.frequency_mhz))
            selection_path = 'energy_min'
        else:
            # The repository returns every global minimum-TTFT exact tie.
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
            t0_ns=t0_ns, tp_ns=tp_ns, prefilter_ns=tp_ns,
            selection_rule='reference_budget', lookup_end_ns=lookup_end_ns)
