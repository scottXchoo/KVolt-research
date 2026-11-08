"""Manual PC choices, phase costs and rejection of incomplete foreign inputs."""
from dataclasses import FrozenInstanceError, replace
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy.prefill import PrefillController
from kvolt_runtime.policy.types import (
    Bracket, FREQUENCIES_MHZ, PrefillLatencyPrediction, PrefillLookup, PrefillPrediction,
)


class Repo:
    def __init__(self, target=2.0):
        self.calls = []
        self.target = target
        self.latency_rows = None
        self.detail_rows = None
        self.lookup = None
        self.rows = tuple(PrefillPrediction(1024, f, 1.0, 10.0, Bracket(1024, 1024, 0))
                          for f in FREQUENCIES_MHZ)

    def prefill(self, length):
        pytest.fail('PC used the offline full-details API')

    def prefill_lookup(self, length, *, latency_only=False):
        self.calls.append(('lookup', length, latency_only))
        assert latency_only is True
        if self.lookup is not None:
            return self.lookup
        # Bypass construction to exercise validation of foreign/corrupted data.
        lookup = object.__new__(PrefillLookup)
        for name, value in (('length', length), ('predictions', self.latencies()),
                            ('target_s', self.target), ('bracket', Bracket(1024, 1024, 0))):
            object.__setattr__(lookup, name, value)
        return lookup

    def prefill_latencies(self, length):
        pytest.fail('PC used the separate latency API')

    def latencies(self):
        if self.latency_rows is not None:
            return self.latency_rows
        rows = []
        for row in self.rows:
            # Preserve foreign mutations for controller-side revalidation.
            prediction = object.__new__(PrefillLatencyPrediction)
            for field in ('length', 'frequency_mhz', 'ttft_s', 'bracket'):
                object.__setattr__(prediction, field, getattr(row, field))
            object.__setattr__(prediction, 'energy_j', None)
            rows.append(prediction)
        return tuple(rows)

    def prefill_candidates(self, length, frequencies):
        self.calls.append(('details', length, frequencies))
        if self.detail_rows is not None:
            return self.detail_rows
        return tuple(row for row in self.rows if row.frequency_mhz in frequencies)

    def prefill_target(self, length):
        pytest.fail('PC used the separate target API')

    def cell(self, frequency, **changes):
        self.rows = tuple(replace(row, **changes) if row.frequency_mhz == frequency else row
                          for row in self.rows)


def choose(repo, *, elapsed=0, t0=100):
    return PrefillController(repo, clock=lambda: t0 + elapsed).choose(1024, t0)


def test_invalid_combined_target_prevents_details():
    repo = Repo(target=math.nan)
    with pytest.raises(ContractError):
        choose(repo)
    assert repo.calls == [('lookup', 1024, True)]


def test_prefilter_looks_up_energy_only_inside_remaining_time_budget():
    repo = Repo(target=2)
    repo.cell(408, ttft_s=1.6, energy_j=0)
    repo.cell(510, ttft_s=math.nextafter(1.5, math.inf), energy_j=0)
    repo.cell(612, ttft_s=1.5, energy_j=1)
    result = choose(repo, elapsed=500_000_000)
    expected = tuple(f for f in FREQUENCIES_MHZ if f not in (408, 510))
    assert repo.calls == [('lookup', 1024, True), ('details', 1024, expected)]
    assert result.selected.frequency_mhz == 612
    assert all(c.prediction.energy_j is None for c in result.candidates[:2])
    assert all(isinstance(c.prediction, PrefillLatencyPrediction) for c in result.candidates[:2])
    assert all(isinstance(c.prediction, PrefillPrediction) for c in result.candidates[2:])


def test_detail_lookup_cost_rechecks_final_budget_and_changes_energy_winner():
    repo = Repo(target=2)
    repo.cell(408, ttft_s=1.25, energy_j=0)
    repo.cell(510, ttft_s=1.0, energy_j=1)
    clocks = iter((500_000_100, 1_000_000_100))
    result = PrefillController(repo, clock=lambda: next(clocks)).choose(1024, 100)
    assert result.prefilter_ns == 500_000_100
    assert (result.tp_ns, result.elapsed_ns, result.budget_s) == (1_000_000_100, 1_000_000_000, 1)
    assert result.selected.frequency_mhz == 510
    assert result.candidates[0].prediction.energy_j == 0
    assert result.candidates[0].reasons == ('time_budget_exceeded', 'not_selected')


@pytest.mark.parametrize('initial_empty', [False, True])
def test_fallback_fetches_every_global_fastest_tie_and_keeps_energy_frequency_order(initial_empty):
    repo = Repo(target=0.1 if initial_empty else 1)
    repo.cell(408, ttft_s=0.5, energy_j=3)
    repo.cell(816, ttft_s=0.5, energy_j=2)
    repo.cell(1122, ttft_s=0.5, energy_j=2)
    clocks = iter((100, 1_000_000_100))
    result = PrefillController(repo, clock=lambda: next(clocks)).choose(1024, 100)
    expected = (408, 816, 1122) if initial_empty else FREQUENCIES_MHZ
    assert repo.calls[-1] == ('details', 1024, expected)
    assert result.selected.frequency_mhz == 816
    assert result.selection_path == 'latency_fallback'
    assert result.predicted_feasible is False
    assert result.budget_s == repo.target - 1
    assert all(not c.meets_budget for c in result.candidates)


def test_clock_cannot_move_backwards_between_prefilter_and_final_lookup():
    clocks = iter((200, 199))
    with pytest.raises(ContractError, match='backwards'):
        PrefillController(Repo(), clock=lambda: next(clocks)).choose(1024, 100)


@pytest.mark.parametrize('bad', [True, False, None, '200', 200.0, -1, 99])
def test_final_lookup_clock_must_be_valid_after_detail_work(bad):
    repo = Repo()
    clocks = iter((100, bad))
    with pytest.raises(ContractError):
        PrefillController(repo, clock=lambda: next(clocks)).choose(1024, 100)
    assert repo.calls[-1] == ('details', 1024, FREQUENCIES_MHZ)


@pytest.mark.parametrize('kind', ['list', 'empty', 'missing', 'duplicate', 'extra',
                                 'unrequested', 'foreign', 'latency_only', 'wrong_length',
                                 'different_ttft', 'different_bracket'])
def test_foreign_detail_subset_must_match_requested_rows_and_prior_latency(kind):
    repo = Repo(target=0.75)
    repo.cell(510, ttft_s=0.5)
    repo.cell(1122, ttft_s=0.5)
    rows = (repo.rows[1], repo.rows[-1])
    repo.detail_rows = {
        'list': list(rows), 'empty': (), 'missing': rows[:-1],
        'duplicate': (rows[0], rows[0]), 'extra': rows + (repo.rows[0],),
        'unrequested': (repo.rows[0], rows[1]),
        'foreign': (object(), rows[1]),
        'latency_only': (PrefillLatencyPrediction(1024, 510, .5, Bracket(1024, 1024, 0)), rows[1]),
        'wrong_length': (PrefillPrediction(2048, 510, .5, 10, Bracket(2048, 2048, 0)), rows[1]),
        'different_ttft': (replace(rows[0], ttft_s=math.nextafter(.5, math.inf)), rows[1]),
        'different_bracket': (replace(rows[0], bracket=Bracket(1024, 2048, 0)), rows[1]),
    }[kind]
    with pytest.raises(ContractError):
        choose(repo)
    assert repo.calls[-1] == ('details', 1024, (510, 1122))


@pytest.mark.parametrize('field,bad', [('energy_j', 0), ('energy_j', math.nan),
                                      ('frequency_mhz', True), ('ttft_s', math.nan),
                                      ('length', True), ('bracket', None)])
def test_foreign_latency_dataclass_is_revalidated_before_details(field, bad):
    repo = Repo()
    repo.latency_rows = repo.latencies()
    repo.calls.clear()
    object.__setattr__(repo.latency_rows[0], field, bad)
    with pytest.raises(ContractError):
        choose(repo)
    assert repo.calls == [('lookup', 1024, True)]


def test_energy_minimum_filters_fast_expensive_and_slow_cheap_candidates():
    repo = Repo()
    repo.cell(408, ttft_s=3.0, energy_j=0.0)
    repo.cell(510, ttft_s=0.1, energy_j=100.0)
    repo.cell(816, ttft_s=1.5, energy_j=2.0)
    result = choose(repo)
    assert result.selected.frequency_mhz == 816
    assert result.selection_path == 'energy_min' and result.predicted_feasible is True
    assert (result.length, result.target_s, result.budget_s) == (1024, 2.0, 2.0)
    assert (result.t0_ns, result.tp_ns, result.elapsed_ns) == (100, 100, 0)
    assert len(result.candidates) == 8 and type(result.candidates) is tuple
    assert [c.prediction.frequency_mhz for c in result.candidates] == list(FREQUENCIES_MHZ)
    assert sum(c.selected for c in result.candidates) == 1
    assert result.candidates[0].reasons == ('time_budget_exceeded', 'not_selected')
    assert result.candidates[1].reasons == ('not_selected',)
    assert next(c for c in result.candidates if c.selected).reasons == ()
    for record in (result, result.selected, result.candidates[0]):
        with pytest.raises(FrozenInstanceError):
            record.length = 2048


@pytest.mark.parametrize('elapsed,expected_budget', [(0, 0.1), (200_000_000, -0.1)])
def test_no_feasible_candidate_uses_global_latency_and_preserves_budget(elapsed, expected_budget):
    repo = Repo(target=0.1)
    repo.cell(1122, ttft_s=0.2, energy_j=100)
    repo.cell(408, ttft_s=0.3, energy_j=0)
    result = choose(repo, elapsed=elapsed)
    assert result.budget_s == expected_budget
    assert result.selected.frequency_mhz == 1122
    assert result.selection_path == 'latency_fallback'
    assert result.predicted_feasible is False
    assert all(not c.meets_budget for c in result.candidates)
    selected = next(c for c in result.candidates if c.selected)
    assert selected.reasons == ('time_budget_exceeded', 'latency_fallback')


@pytest.mark.parametrize('fallback', [False, True])
def test_exact_ties_use_priority_order_independent_of_repository_row_order(fallback):
    repo = Repo(target=0.1 if fallback else 2)
    # Normal: equal energy chooses lower TTFT, then lower f.
    # Fallback: equal TTFT chooses lower energy, then lower f.
    repo.cell(510, ttft_s=0.5, energy_j=2)
    repo.cell(612, ttft_s=0.5, energy_j=2)
    repo.cell(408, ttft_s=0.8 if not fallback else 0.5, energy_j=2 if not fallback else 3)
    repo.rows = tuple(reversed(repo.rows))
    assert choose(repo).selected.frequency_mhz == 510


def test_tiny_energy_difference_is_not_an_approximate_tie():
    repo = Repo()
    repo.cell(408, ttft_s=0.1, energy_j=math.nextafter(1.0, math.inf))
    repo.cell(1122, ttft_s=1.0, energy_j=1.0)
    assert choose(repo).selected.frequency_mhz == 1122


@pytest.mark.parametrize('fallback', [False, True])
def test_tiny_time_difference_is_not_an_approximate_tie(fallback):
    repo = Repo(target=0.1 if fallback else 2)
    repo.cell(408, ttft_s=math.nextafter(0.5, math.inf), energy_j=1)
    repo.cell(1122, ttft_s=0.5, energy_j=1 if not fallback else 100)
    assert choose(repo).selected.frequency_mhz == 1122


def test_equal_budget_is_feasible_but_next_binary64_value_is_not():
    repo = Repo(target=1)
    repo.cell(408, ttft_s=math.nextafter(0.5, math.inf), energy_j=0)
    repo.cell(1122, ttft_s=0.5, energy_j=1)
    result = choose(repo, elapsed=500_000_000)
    assert result.budget_s == 0.5
    assert result.selected.frequency_mhz == 1122
    assert result.candidates[0].meets_budget is False
    assert result.candidates[-1].meets_budget is True


def test_phase_order_measures_lookup_and_target_cost_and_never_calls_other_phases():
    repo = Repo()
    def forbidden(*args):
        pytest.fail('PC called preflight, Quality, Decode or Knee')
    repo.preflight = repo.quality = repo.decode = repo.decode_lookup = repo.decode_target = repo.detect = forbidden
    def clock():
        repo.calls.append(('clock', None))
        return 500_000_100
    result = PrefillController(repo, clock=clock).choose(1024, 100)
    assert repo.calls == [('lookup', 1024, True), ('clock', None),
                          ('details', 1024, FREQUENCIES_MHZ), ('clock', None)]
    assert result.prefilter_ns == 500_000_100
    assert result.elapsed_ns == 500_000_000 and result.budget_s == 1.5


def test_integer_subtraction_preserves_nanoseconds_after_huge_clock_origin():
    t0 = 10**100
    result = choose(Repo(), elapsed=17, t0=t0)
    assert result.elapsed_ns == 17 and type(result.elapsed_ns) is int
    assert result.budget_s == 2 - 17 / 1_000_000_000
    assert (result.t0_ns, result.tp_ns) == (t0, t0 + 17)


def test_repeated_requests_repeat_online_lookup_and_target_no_cached_decision():
    repo = Repo()
    controller = PrefillController(repo, clock=lambda: 100)
    assert controller.choose(1024, 100).selected.frequency_mhz == 408
    repo.cell(1122, energy_j=0)
    repo.target = 0.1
    result = controller.choose(1024, 100)
    assert result.selection_path == 'latency_fallback' and result.selected.frequency_mhz == 1122
    assert repo.calls == [('lookup', 1024, True),
                          ('details', 1024, FREQUENCIES_MHZ)] * 2


@pytest.mark.parametrize('bad', [True, False, None, '1024', 1024.0, 1023, 8193])
def test_invalid_length_rejected_before_repository(bad):
    repo = Repo()
    with pytest.raises(ContractError):
        PrefillController(repo, clock=lambda: 100).choose(bad, 100)
    assert repo.calls == []


@pytest.mark.parametrize('bad', [True, False, None, '0', 0.0, -1])
def test_invalid_phase_start_rejected_before_repository(bad):
    repo = Repo()
    with pytest.raises(ContractError):
        PrefillController(repo, clock=lambda: 100).choose(1024, bad)
    assert repo.calls == []


@pytest.mark.parametrize('bad', [True, False, None, '100', 100.0, -1, 99])
def test_invalid_or_backward_clock_is_fatal(bad):
    with pytest.raises(ContractError):
        PrefillController(Repo(), clock=lambda: bad).choose(1024, 100)


@pytest.mark.parametrize('bad', [None, True, '2', 0, -1, math.nan, math.inf, -math.inf])
def test_invalid_target_is_fatal_and_does_not_read_clock(bad):
    def clock():
        pytest.fail('invalid target reached phase clock')
    with pytest.raises(ContractError):
        PrefillController(Repo(target=bad), clock=clock).choose(1024, 100)


@pytest.mark.parametrize('kind', ['list', 'empty', 'missing', 'duplicate', 'extra', 'foreign', 'wrong_length'])
def test_incomplete_or_foreign_rows_rejected(kind):
    repo = Repo()
    rows = repo.latencies()
    repo.calls.clear()
    repo.latency_rows = {
        'list': list(rows), 'empty': (), 'missing': rows[:-1],
        'duplicate': rows[:-1] + (rows[0],), 'extra': rows + (rows[0],),
        'foreign': (object(),) + rows[1:],
        'wrong_length': (PrefillLatencyPrediction(2048, 408, 1, Bracket(2048, 2048, 0)),) + rows[1:],
    }[kind]
    with pytest.raises(ContractError):
        choose(repo)
    assert repo.calls == [('lookup', 1024, True)]


@pytest.mark.parametrize('field,bad', [
    ('length', True), ('frequency_mhz', 409), ('frequency_mhz', 408.0),
    ('ttft_s', 0), ('ttft_s', math.nan), ('ttft_s', math.inf), ('ttft_s', True),
    ('energy_j', -1), ('energy_j', math.nan), ('energy_j', math.inf), ('energy_j', '0'),
    ('bracket', None), ('bracket', Bracket(2048, 2048, 0)),
])
def test_corrupted_prediction_dataclass_is_revalidated(field, bad):
    repo = Repo()
    object.__setattr__(repo.rows[0], field, bad)
    with pytest.raises(ContractError):
        choose(repo)


@pytest.mark.parametrize('field,bad', [('lower', 1024.0), ('upper', 8193), ('weight', math.nan)])
def test_corrupted_nested_bracket_is_revalidated(field, bad):
    repo = Repo()
    object.__setattr__(repo.rows[0].bracket, field, bad)
    with pytest.raises(ContractError):
        choose(repo)


@pytest.mark.parametrize('field,fallback', [('energy_j', False), ('ttft_s', False), ('ttft_s', True)])
def test_foreign_integer_metrics_are_normalized_before_binary64_comparison(field, fallback):
    base = 2**53
    repo = Repo(target=1 if fallback else 2**54)
    repo.rows = tuple(replace(row, **{field: float(base + 2)}) for row in repo.rows)
    repo.cell(510, **{field: float(base)})
    object.__setattr__(repo.rows[0], field, base + 1)
    # float(2**53+1) == float(2**53); canonical binary64 metrics tie.
    # The original 408 MHz identifier wins their final frequency tie.
    result = choose(repo)
    assert result.selected.frequency_mhz == 408
    assert getattr(result.selected, field) == float(base)
    assert type(getattr(result.selected, field)) is float


@pytest.mark.parametrize('length', [1024, 1025, 1536, 8192])
def test_mock_supports_knots_interpolation_and_boundaries(length):
    from kvolt_runtime.policy.mock import MockLUTRepository
    result = PrefillController(MockLUTRepository('exact_ties'), clock=lambda: 0).choose(length, 0)
    assert result.length == length and result.selected.frequency_mhz == 408
    assert len(result.candidates) == 8


def test_import_and_choose_require_no_numerical_or_gpu_library():
    code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'kneed', 'torch', 'transformers'}:
            raise AssertionError('forbidden policy dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.prefill import PrefillController
assert PrefillController(MockLUTRepository(), clock=lambda: 0).choose(1024, 0).selected
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('length,elapsed,target,time,energy,path', [
    (1024, 0, 0.80568193, 0.7636938302, 22.95247488752233, 'energy_min'),
    (1536, 0, 0.5 * 0.80568193 + 0.5 * 1.974938814,
     0.5 * 0.7636938302 + 0.5 * 1.9121312994,
     0.5 * 22.95247488752233 + 0.5 * 70.02650760043105, 'energy_min'),
    (2048, 0, 1.974938814, 1.9121312994, 70.02650760043105, 'energy_min'),
    (2048, 100_000_000, 1.974938814, 1.9121312994, 70.02650760043105, 'latency_fallback'),
])
def test_sealed_bundle_manual_small_cases(length, elapsed, target, time, energy, path):
    from kvolt_runtime.policy.lut import MeasuredLUTRepository
    configured = os.environ.get('KVOLT_LUT_BUNDLE')
    bundle = Path(configured) if configured else Path(__file__).resolve().parents[2] / 'KVolt-lut-source/data/luts/agx2-repeat5-20261103'
    if not bundle.is_dir():
        pytest.skip('sealed measured integration unavailable: set KVOLT_LUT_BUNDLE')
    # Values independently transcribed from the sealed source's 1024/2048
    # knots; 1536 is their midpoint. All eight candidates remain visible.
    # At 2048 the 1020 MHz row uses less energy (69.42603514426558 J),
    # but its 2.0801904404 s TTFT exceeds the 1.974938814 s reference.
    result = PrefillController(MeasuredLUTRepository(bundle), clock=lambda: elapsed).choose(length, 0)
    assert result.selected.frequency_mhz == 1122
    assert (result.target_s, result.selected.ttft_s, result.selected.energy_j) == (target, time, energy)
    assert result.budget_s == target - elapsed / 1_000_000_000
    assert result.selection_path == path
    assert result.predicted_feasible is (path == 'energy_min')
    assert len(result.candidates) == 8
    if length == 2048:
        lower_energy = next(c for c in result.candidates if c.prediction.frequency_mhz == 1020)
        assert lower_energy.prediction.energy_j is None
        assert lower_energy.prediction.ttft_s == 2.0801904404
        original = next(row for row in MeasuredLUTRepository(bundle).prefill(length)
                        if row.frequency_mhz == 1020)
        assert original.energy_j == 69.42603514426558
        assert lower_energy.meets_budget is False


@pytest.mark.parametrize('field,bad', [
    ('length', True), ('length', 2048), ('bracket', None),
    ('bracket', Bracket(2048, 2048, 0)),
    ('bracket', Bracket(1024, 2048, 0)),
])
def test_combined_lookup_metadata_is_validated_before_details_and_clock(field, bad):
    repo = Repo()
    repo.lookup = repo.prefill_lookup(1024, latency_only=True)
    repo.calls.clear()
    object.__setattr__(repo.lookup, field, bad)
    with pytest.raises(ContractError):
        PrefillController(repo, clock=lambda: pytest.fail('invalid lookup reached clock')).choose(1024, 100)
    assert repo.calls == [('lookup', 1024, True)]


@pytest.mark.parametrize('bad', [object(), None, {}, ()])
def test_foreign_combined_lookup_wrapper_is_rejected(bad):
    repo = Repo()
    repo.prefill_lookup = lambda *args, **kwargs: bad
    with pytest.raises(ContractError):
        PrefillController(repo, clock=lambda: pytest.fail('invalid wrapper reached clock')).choose(1024, 100)


@pytest.mark.parametrize('field,bad', [('lower', 1024.0), ('upper', 8193), ('weight', math.nan)])
def test_corrupted_combined_lookup_bracket_is_revalidated(field, bad):
    repo = Repo()
    repo.lookup = repo.prefill_lookup(1024, latency_only=True)
    object.__setattr__(repo.lookup.bracket, field, bad)
    with pytest.raises(ContractError):
        choose(repo)
