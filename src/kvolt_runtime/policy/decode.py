"""CPU-only Decode selection at the first-token phase boundary."""
import time

from ..config import (EOS_TOKEN_IDS, GROUPS, MISSING, buffered_tpot_target,
                      require_int, resolve_tpot_buffer_ratio)
from ..errors import ContractError
from .knee import KneeAdapter, validate_quality
from .types import (
    Bracket, DecodeCandidate, DecodeDecision, DecodePrediction, FREQUENCIES_MHZ,
    KneeResult, LUTRepository, elapsed_since, length as validate_length,
)


def _validate_decode(predictions, requested_length):
    expected = {(g, f) for g in GROUPS for f in FREQUENCIES_MHZ}
    if type(predictions) is not tuple or not 1 <= len(predictions) <= len(expected):
        raise ContractError('Decode budget query must return an immutable tuple of 1..40 rows')
    normalized = []
    for row in predictions:
        if not isinstance(row, DecodePrediction):
            raise ContractError('Decode LUT must return DecodePrediction rows')
        if not isinstance(row.bracket, Bracket):
            raise ContractError('Decode prediction requires an interpolation bracket')
        # Use canonical copies for selection and traces, not the foreign source.
        bracket = Bracket(row.bracket.lower, row.bracket.upper, row.bracket.weight)
        prediction = DecodePrediction(row.length, row.group, row.frequency_mhz, row.tpot_s,
                                      row.energy_j_per_token, bracket)
        if prediction.length != requested_length:
            raise ContractError('Decode LUT rows must match requested L')
        normalized.append(prediction)
    pairs = {(row.group, row.frequency_mhz) for row in normalized}
    if len(pairs) != len(normalized) or not pairs <= expected:
        raise ContractError('Decode budget query must return unique original (G,f) pairs')
    if any(row.bracket != normalized[0].bracket for row in normalized):
        raise ContractError('Decode budget query rows must share an interpolation bracket')
    return tuple(sorted(normalized, key=lambda row: (row.group, row.frequency_mhz)))


class DecodeController:
    """Query Q/Knees and Reference, record td/budget, then query Decode details.

    Runner preflight remains outside this online phase. This controller neither
    caches per-request interpolation/detection nor changes device state.
    """

    def __init__(self, repo: LUTRepository, knee_adapter=None, *, clock=time.monotonic_ns,
                 tpot_buffer_ratio=MISSING):
        self._tpot_buffer_ratio = resolve_tpot_buffer_ratio(tpot_buffer_ratio)
        self.repo = repo
        self.knee_adapter = KneeAdapter() if knee_adapter is None else knee_adapter
        self.clock = clock

    @property
    def tpot_buffer_ratio(self):
        """Policy snapshot fixed for this controller/session."""
        return self._tpot_buffer_ratio

    def choose(self, length: int, nmax: int, first_token_id: int, t1_ns: int) -> DecodeDecision:
        validate_length(length)
        require_int(nmax, 'Nmax', 1, 512)
        require_int(first_token_id, 'first token ID', 0, 128255)
        require_int(t1_ns, 'first token ns', 0)

        if first_token_id in EOS_TOKEN_IDS or nmax == 1:
            reason = 'first_token_eos' if first_token_id in EOS_TOKEN_IDS else 'nmax_one'
            return DecodeDecision(length, nmax, first_token_id, 'skipped', 'skipped',
                                  t1_ns=t1_ns, skip_reason=reason)

        quality = validate_quality(self.repo.quality(length), length)
        knee = self.knee_adapter.detect(quality)
        if not isinstance(knee, KneeResult):
            raise ContractError('Knee adapter must return KneeResult')
        KneeResult(knee.status, knee.groups, knee.reason, knee.error_type, knee.error_message)
        if knee.status == 'found':
            active = set(knee.groups)
        elif knee.status == 'not_found':
            active = {sorted(GROUPS)[len(GROUPS) // 2]}
        else:
            active = set(GROUPS)
        reference_target = self.repo.decode_target(length)
        target = buffered_tpot_target(reference_target, self.tpot_buffer_ratio)
        td_ns = self.clock()
        elapsed_ns = elapsed_since(t1_ns, td_ns)
        budget = target - (elapsed_ns / 1_000_000_000) / (nmax - 1)
        predictions = _validate_decode(
            self.repo.decode_within_budget(length, tuple(sorted(active)), budget), length)
        scores = {row.group: row.score for row in quality}
        eligible = tuple(row for row in predictions if row.group in active and row.tpot_s <= budget)
        if eligible:
            if len(eligible) != len(predictions):
                raise ContractError('Decode budget query mixed eligible and ineligible rows')
            selected = min(eligible, key=lambda row: (
                row.energy_j_per_token, -scores[row.group], row.tpot_s,
                row.group, row.frequency_mhz))
            selection_path = 'energy_min'
        else:
            if any(row.tpot_s != predictions[0].tpot_s for row in predictions):
                raise ContractError('Decode fallback query must return only exact fastest TPOT ties')
            selected = min(predictions, key=lambda row: (
                row.tpot_s, -scores[row.group], row.energy_j_per_token,
                row.group, row.frequency_mhz))
            selection_path = 'latency_fallback'

        candidates = []
        for row in predictions:
            active_knee = row.group in active
            meets_budget = row.tpot_s <= budget
            is_selected = row is selected
            reasons = []
            if not active_knee:
                reasons.append('knee_excluded')
            if not meets_budget:
                reasons.append('time_budget_exceeded')
            if not is_selected:
                reasons.append('not_selected')
            elif selection_path == 'latency_fallback':
                reasons.append('latency_fallback')
            candidates.append(DecodeCandidate(
                row, scores[row.group], active_knee, meets_budget,
                active_knee and meets_budget, is_selected, tuple(reasons)))

        return DecodeDecision(
            length=length, nmax=nmax, first_token_id=first_token_id,
            status='selected', selection_path=selection_path, selected=selected,
            selected_quality=scores[selected.group], target_s=target, budget_s=budget,
            elapsed_ns=elapsed_ns, predicted_feasible=selected.tpot_s <= budget,
            quality=quality, knee=knee, candidates=tuple(candidates), t1_ns=t1_ns, td_ns=td_ns,
            reference_target_s=float(reference_target), tpot_buffer_ratio=self.tpot_buffer_ratio)
