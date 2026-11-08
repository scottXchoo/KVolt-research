"""CPU-only Prefill energy selection inside the online PC phase."""
import time

from ..config import require_int
from ..errors import ContractError
from .types import (
    Bracket, FREQUENCIES_MHZ, LUTRepository, PrefillCandidate, PrefillDecision,
    PrefillLatencyPrediction, PrefillLookup, PrefillPrediction, elapsed_since,
    length as validate_length,
)


def _validate_prefill(predictions, requested_length, frequencies, *, latency_only=False):
    if type(predictions) is not tuple or len(predictions) != len(frequencies):
        raise ContractError('Prefill LUT must return an immutable tuple of the requested frequencies')
    prediction_type = PrefillLatencyPrediction if latency_only else PrefillPrediction
    normalized = []
    for row in predictions:
        if not isinstance(row, prediction_type):
            raise ContractError(f'Prefill LUT must return {prediction_type.__name__} rows')
        if not isinstance(row.bracket, Bracket):
            raise ContractError('Prefill prediction requires an interpolation bracket')
        # Revalidate even frozen dataclasses supplied by a foreign repository.
        bracket = Bracket(row.bracket.lower, row.bracket.upper, row.bracket.weight)
        if latency_only:
            prediction = PrefillLatencyPrediction(
                row.length, row.frequency_mhz, row.ttft_s, bracket, row.energy_j)
        else:
            prediction = PrefillPrediction(
                row.length, row.frequency_mhz, row.ttft_s, row.energy_j, bracket)
        if row.length != requested_length:
            raise ContractError('Prefill LUT rows must match requested L')
        normalized.append(prediction)
    if {row.frequency_mhz for row in normalized} != set(frequencies):
        raise ContractError('Prefill LUT must contain every requested frequency exactly once')
    return tuple(sorted(normalized, key=lambda row: row.frequency_mhz))


class PrefillController:
    """Query target and all TTFTs together before budget-eligible details.

    The M3 caller performs full-support preflight before entering this phase.
    This controller does no device/file work or request-result caching.
    """

    def __init__(self, repo: LUTRepository, *, clock=time.monotonic_ns):
        self.repo = repo
        self.clock = clock

    def choose(self, length: int, t0_ns: int) -> PrefillDecision:
        validate_length(length)
        require_int(t0_ns, 'Prefill start ns', 0)

        lookup = self.repo.prefill_lookup(length, latency_only=True)
        if not isinstance(lookup, PrefillLookup):
            raise ContractError('Prefill LUT must return PrefillLookup')
        if not isinstance(lookup.bracket, Bracket):
            raise ContractError('Prefill lookup requires an interpolation bracket')
        bracket = Bracket(lookup.bracket.lower, lookup.bracket.upper, lookup.bracket.weight)
        lookup = PrefillLookup(lookup.length, lookup.predictions, lookup.target_s, bracket)
        if lookup.length != length:
            raise ContractError('Prefill lookup must match requested L')
        target = lookup.target_s
        latencies = _validate_prefill(
            lookup.predictions, length, FREQUENCIES_MHZ, latency_only=True)
        if any(row.bracket != bracket for row in latencies):
            raise ContractError('Prefill rows must match the lookup interpolation bracket')
        prefilter_ns = self.clock()
        prefilter_elapsed_ns = elapsed_since(t0_ns, prefilter_ns)
        prefilter_budget = target - prefilter_elapsed_ns / 1_000_000_000
        shortlist = tuple(row.frequency_mhz for row in latencies if row.ttft_s <= prefilter_budget)
        if not shortlist:
            fastest = min(row.ttft_s for row in latencies)
            shortlist = tuple(row.frequency_mhz for row in latencies if row.ttft_s == fastest)

        predictions = _validate_prefill(
            self.repo.prefill_candidates(length, shortlist), length, shortlist)
        latency_by_frequency = {row.frequency_mhz: row for row in latencies}
        for row in predictions:
            prior = latency_by_frequency[row.frequency_mhz]
            if row.ttft_s != prior.ttft_s or row.bracket != prior.bracket:
                raise ContractError('Prefill details must match the prior TTFT and interpolation bracket')
        tp_ns = self.clock()
        elapsed_since(prefilter_ns, tp_ns)
        elapsed_ns = elapsed_since(t0_ns, tp_ns)
        budget = target - elapsed_ns / 1_000_000_000

        feasible = tuple(row for row in predictions if row.ttft_s <= budget)
        if feasible:
            selected = min(feasible, key=lambda row: (row.energy_j, row.ttft_s, row.frequency_mhz))
            selection_path = 'energy_min'
        else:
            # Every global fastest tie is in any nonempty time prefix, or
            # explicitly fetched when the initial time prefix was empty.
            selected = min(predictions, key=lambda row: (row.ttft_s, row.energy_j, row.frequency_mhz))
            selection_path = 'latency_fallback'

        candidates = []
        detail_by_frequency = {row.frequency_mhz: row for row in predictions}
        for latency in latencies:
            row = detail_by_frequency.get(latency.frequency_mhz, latency)
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
            t0_ns=t0_ns, tp_ns=tp_ns, prefilter_ns=prefilter_ns)
