"""Manual controller scenarios independently specify the chosen pair."""
from dataclasses import replace
import math

import pytest

from kvolt_runtime.config import GROUPS, EOS_TOKEN_IDS
from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy.decode import DecodeController
from kvolt_runtime.policy.knee import KneeAdapter
from kvolt_runtime.policy.types import (
    Bracket, DecodePrediction, FREQUENCIES_MHZ, KneeResult, QualityPrediction,
)


class Repo:
    def __init__(self, scores=(0.9, 0.8, 0.7, 0.6, 0.5), target=2.0):
        self.calls = []
        self.target = target
        self.qualities = tuple(QualityPrediction(1024, g, q, Bracket(1024, 1024, 0))
                               for g, q in zip(GROUPS, scores))
        self.rows = tuple(DecodePrediction(1024, g, f, 1.0, 10.0, Bracket(1024, 1024, 0))
                          for g in GROUPS for f in FREQUENCIES_MHZ)

    def quality(self, length):
        self.calls.append('quality')
        return self.qualities

    def decode(self, length):
        self.calls.append('decode')
        return self.rows

    def decode_target(self, length):
        self.calls.append('target')
        return self.target

    def cell(self, group, frequency=408, **changes):
        self.rows = tuple(replace(row, **changes) if (row.group, row.frequency_mhz) ==
                          (group, frequency) else row for row in self.rows)


class Adapter:
    def __init__(self, result=None, events=None):
        self.result = result or KneeResult('not_found', reason='no_knee')
        self.events = events

    def detect(self, quality):
        if self.events is not None:
            self.events.append('knee')
        return self.result


def choose(repo, knee=None, *, elapsed=0, nmax=128, t1=100):
    return DecodeController(repo, knee or Adapter(), clock=lambda: t1 + elapsed).choose(
        1024, nmax, 123, t1)


def pair(decision):
    return decision.selected.group, decision.selected.frequency_mhz


def test_normal_energy_choice_and_full_explanation():
    repo = Repo()
    repo.cell(16, 612, energy_j_per_token=2.0)
    result = choose(repo, Adapter(KneeResult('found', (8, 16))))
    assert pair(result) == (16, 612)
    assert result.selected_quality == 0.8
    assert result.status == 'selected' and result.selection_path == 'energy_min'
    assert result.predicted_feasible is True
    assert len(result.candidates) == 40 and len(result.quality) == 5
    assert sum(c.selected for c in result.candidates) == 1
    outside = next(c for c in result.candidates if c.prediction.group == 128)
    assert (outside.active_knee, outside.meets_budget, outside.eligible) == (False, True, False)
    assert 'knee_excluded' in outside.reasons


@pytest.mark.parametrize('status', ['not_found', 'error_recovered'])
def test_not_found_and_recovery_apply_time_condition_to_all_groups(status):
    repo = Repo()
    repo.cell(128, 1122, energy_j_per_token=0.1)
    result = choose(repo, Adapter(KneeResult(status, reason='test')))
    assert pair(result) == (128, 1122)
    assert all(c.active_knee for c in result.candidates)
    assert result.knee.status == status


def test_empty_knee_time_set_releases_knee_and_fallback_can_still_meet_budget():
    repo = Repo(target=0.5)
    repo.cell(128, 1020, tpot_s=0.2, energy_j_per_token=100)
    result = choose(repo, Adapter(KneeResult('found', (8,))))
    assert pair(result) == (128, 1020)
    assert result.selection_path == 'latency_fallback'
    assert result.predicted_feasible is True
    selected = next(c for c in result.candidates if c.selected)
    assert selected.eligible is False and selected.active_knee is False
    assert 'latency_fallback' in selected.reasons


def test_negative_budget_kept_and_global_latency_fallback():
    repo = Repo(target=0.1)
    repo.cell(32, 816, tpot_s=0.2)
    result = choose(repo, elapsed=2_000_000_000, nmax=2)
    assert result.budget_s == -1.9
    assert pair(result) == (32, 816)
    assert result.predicted_feasible is False
    assert all(not c.meets_budget for c in result.candidates)
    assert all('time_budget_exceeded' in c.reasons for c in result.candidates)


@pytest.mark.parametrize('nmax', [2, 128, 512])
def test_budget_uses_nmax_minus_one_and_integer_clock_subtraction(nmax):
    t1 = 10**25  # Beyond binary64's exact integer range.
    result = choose(Repo(target=2), elapsed=1_000_000_001, nmax=nmax, t1=t1)
    assert result.elapsed_ns == 1_000_000_001
    assert result.budget_s == 2 - (1_000_000_001 / 1_000_000_000) / (nmax - 1)
    assert result.td_ns == t1 + 1_000_000_001


def test_phase_order_and_detector_failure_cost_included_in_clock():
    repo = Repo(target=2)
    def broken(**kwargs):
        repo.calls.append('knee')
        raise ValueError('failure consumes control time')
    def clock():
        repo.calls.append('clock')
        return 1_000_000_100
    result = DecodeController(repo, KneeAdapter(broken), clock=clock).choose(1024, 2, 123, 100)
    assert repo.calls == ['quality', 'knee', 'decode', 'target', 'clock']
    assert result.knee.status == 'error_recovered'
    assert result.elapsed_ns == 1_000_000_000 and result.budget_s == 1.0


@pytest.mark.parametrize('token,nmax,reason', [(123, 1, 'nmax_one')] +
                         [(eos, 128, 'first_token_eos') for eos in EOS_TOKEN_IDS])
def test_skip_makes_no_lookup_detection_or_clock_calls(token, nmax, reason):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('skip must not access repository/adapter')
    def forbidden():
        pytest.fail('skip must not call clock')
    result = DecodeController(Forbidden(), Forbidden(), clock=forbidden).choose(1024, nmax, token, 100)
    assert (result.status, result.selection_path, result.skip_reason) == ('skipped', 'skipped', reason)
    assert result.selected is result.target_s is result.budget_s is result.elapsed_ns is result.td_ns is None
    assert result.quality == () and result.candidates == () and result.knee is None


@pytest.mark.parametrize('args', [(True, 128, 123, 100), (1023, 128, 123, 100),
                                 (8193, 128, 123, 100), (1024, True, 123, 100),
                                 (1024, 513, 123, 100), (1024, 0, 123, 100),
                                 (1024, 1, True, 100), (1024, 1, 128256, 100),
                                 (1024, 1, -1, 100), (1024, 1, 123, 1.0),
                                 (1024, 1, 123, -1)])
def test_invalid_inputs_fail_even_before_skip(args):
    repo = Repo()
    with pytest.raises(ContractError):
        DecodeController(repo, Adapter()).choose(*args)
    assert repo.calls == []


@pytest.mark.parametrize('now', [99, True, 100.0, None])
def test_invalid_or_backward_clock_is_fatal(now):
    with pytest.raises(ContractError):
        DecodeController(Repo(), Adapter(), clock=lambda: now).choose(1024, 128, 123, 100)


@pytest.mark.parametrize('target', [0, -1, float('nan'), float('inf'), True, '1'])
def test_invalid_target_is_fatal_before_clock(target):
    repo = Repo(target=target)
    with pytest.raises(ContractError):
        DecodeController(repo, Adapter(), clock=lambda: pytest.fail('invalid target')).choose(1024, 2, 123, 0)


@pytest.mark.parametrize('mode', ['missing_quality', 'duplicate_quality', 'wrong_quality_length',
                                 'missing_decode', 'duplicate_decode', 'wrong_decode_length', 'corrupt_decode'])
def test_all_five_quality_and_forty_decode_rows_are_mandatory(mode):
    repo = Repo()
    if mode == 'missing_quality':
        repo.qualities = repo.qualities[:-1]
    elif mode == 'duplicate_quality':
        repo.qualities = (repo.qualities[0],) * 5
    elif mode == 'wrong_quality_length':
        repo.qualities = (replace(repo.qualities[0], length=1536, bracket=Bracket(1536, 1536, 0)),) + repo.qualities[1:]
    elif mode == 'missing_decode':
        repo.rows = repo.rows[:-1]
    elif mode == 'duplicate_decode':
        repo.rows = (repo.rows[0],) * 40
    elif mode == 'wrong_decode_length':
        repo.rows = (replace(repo.rows[0], length=1536, bracket=Bracket(1536, 1536, 0)),) + repo.rows[1:]
    else:
        object.__setattr__(repo.rows[-1], 'energy_j_per_token', float('nan'))
    with pytest.raises(ContractError):
        choose(repo)


@pytest.mark.parametrize('fallback', [False, True])
def test_exact_ties_use_quality_then_energy_or_latency_then_group_frequency(fallback):
    repo = Repo(scores=(0.5, 0.9, 0.9, 0.8, 0.7), target=0.1 if fallback else 2)
    # Every primary metric tied; group16 wins Q, then beats group32 by G, f408 by f.
    assert pair(choose(repo)) == (16, 408)
    if fallback:
        repo.cell(32, 510, energy_j_per_token=9)
    else:
        repo.cell(32, 510, tpot_s=0.9)
    assert pair(choose(repo)) == (32, 510)


@pytest.mark.parametrize('fallback', [False, True])
def test_single_binary64_step_beats_quality_tiebreak(fallback):
    repo = Repo(target=0.1 if fallback else 2)
    if fallback:
        repo.cell(128, 1122, tpot_s=math.nextafter(1.0, 0))
    else:
        repo.cell(128, 1122, energy_j_per_token=math.nextafter(10.0, 0))
    assert pair(choose(repo)) == (128, 1122)


def test_budget_boundary_is_exact_and_input_order_does_not_change_choice():
    repo = Repo(target=1)
    repo.cell(16, 510, tpot_s=math.nextafter(1.0, 2), energy_j_per_token=0)
    repo.cell(32, 612, tpot_s=1, energy_j_per_token=1)
    repo.rows = tuple(reversed(repo.rows))
    repo.qualities = tuple(reversed(repo.qualities))
    assert pair(choose(repo)) == (32, 612)


def test_default_adapter_is_set_up_at_controller_construction(monkeypatch):
    import kvolt_runtime.policy.decode as module
    calls = []
    def factory():
        calls.append('startup')
        return Adapter()
    monkeypatch.setattr(module, 'KneeAdapter', factory)
    repo = Repo()
    controller = DecodeController(repo)
    assert calls == ['startup'] and repo.calls == []
    assert controller.choose(1024, 1, 123, 100).status == 'skipped'
    assert repo.calls == []


def test_sealed_published_table_manual_choices_with_explicit_no_knee():
    """Actual data interface check; this test deliberately injects no detector."""
    import os
    from pathlib import Path
    from kvolt_runtime.policy.lut import MeasuredLUTRepository
    configured = os.environ.get('KVOLT_LUT_BUNDLE')
    bundle = (Path(configured) if configured else Path(__file__).resolve().parents[2] /
              'KVolt-lut-source/data/luts/agx2-repeat5-20261103')
    if not bundle.is_dir() and configured is None:
        pytest.skip('sealed published bundle unavailable: set KVOLT_LUT_BUNDLE')
    repo = MeasuredLUTRepository(bundle)
    controller = DecodeController(repo, Adapter(), clock=lambda: 100)
    expected = [(1024, (128, 1122), 'latency_fallback'),
                (1377, (128, 1122), 'latency_fallback'),
                (4096, (128, 1020), 'latency_fallback'),
                (8192, (128, 918), 'energy_min')]
    for length, expected_pair, path in expected:
        result = controller.choose(length, 128, 123, 100)
        assert pair(result) == expected_pair and result.selection_path == path
        assert len(result.candidates) == 40
