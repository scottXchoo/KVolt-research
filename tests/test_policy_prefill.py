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
from kvolt_runtime.policy.types import Bracket, FREQUENCIES_MHZ, PrefillPrediction


class Repo:
    def __init__(self, target=2.0):
        self.calls = []
        self.target = target
        self.rows = tuple(PrefillPrediction(1024, f, 1.0, 10.0, Bracket(1024, 1024, 0))
                          for f in FREQUENCIES_MHZ)

    def prefill(self, length):
        self.calls.append(('prefill', length))
        return self.rows

    def prefill_target(self, length):
        self.calls.append(('target', length))
        return self.target

    def cell(self, frequency, **changes):
        self.rows = tuple(replace(row, **changes) if row.frequency_mhz == frequency else row
                          for row in self.rows)


def choose(repo, *, elapsed=0, t0=100):
    return PrefillController(repo, clock=lambda: t0 + elapsed).choose(1024, t0)


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
    repo.preflight = repo.quality = repo.decode = repo.decode_target = repo.detect = forbidden
    def clock():
        repo.calls.append(('clock', None))
        return 500_000_100
    result = PrefillController(repo, clock=clock).choose(1024, 100)
    assert repo.calls == [('prefill', 1024), ('target', 1024), ('clock', None)]
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
    assert repo.calls == [('prefill', 1024), ('target', 1024)] * 2


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
    repo.rows = {
        'list': list(repo.rows), 'empty': (), 'missing': repo.rows[:-1],
        'duplicate': repo.rows[:-1] + (repo.rows[0],), 'extra': repo.rows + (repo.rows[0],),
        'foreign': (object(),) + repo.rows[1:],
        'wrong_length': (PrefillPrediction(2048, 408, 1, 1, Bracket(2048, 2048, 0)),) + repo.rows[1:],
    }[kind]
    with pytest.raises(ContractError):
        choose(repo)
    assert repo.calls == [('prefill', 1024)]


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
        assert lower_energy.prediction.energy_j == 69.42603514426558
        assert lower_energy.prediction.ttft_s == 2.0801904404
        assert lower_energy.meets_budget is False
