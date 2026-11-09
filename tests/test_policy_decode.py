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


@pytest.fixture(autouse=True)
def historical_strict_tpot_policy(monkeypatch):
    """These manually specified selections cover the historical zero-buffer rule."""
    from kvolt_runtime import config
    monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', 0.0)


class Repo:
    def __init__(self, scores=(0.9, 0.8, 0.7, 0.6, 0.5), target=2.0):
        self.calls = []
        self.target = target
        self.query_result = None
        self.query_args = None
        self.qualities = tuple(QualityPrediction(1024, g, q, Bracket(1024, 1024, 0))
                               for g, q in zip(GROUPS, scores))
        self.rows = tuple(DecodePrediction(1024, g, f, 1.0, 10.0, Bracket(1024, 1024, 0))
                          for g in GROUPS for f in FREQUENCIES_MHZ)

    def quality(self, length):
        self.calls.append('quality')
        return self.qualities

    def decode_lookup(self, length):
        pytest.fail('DC used the full combined Decode API')

    def decode(self, length):
        pytest.fail('DC used the full Decode API')

    def decode_target(self, length):
        self.calls.append('target')
        return self.target

    def decode_within_budget(self, length, groups, budget_s):
        self.calls.append('query')
        self.query_args = (length, groups, budget_s)
        if self.query_result is not None:
            return self.query_result
        eligible = tuple(row for row in self.rows if
                         row.group in groups and float(row.tpot_s) <= budget_s)
        if eligible:
            return eligible
        fastest = min(float(row.tpot_s) for row in self.rows)
        return tuple(row for row in self.rows if float(row.tpot_s) == fastest)

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
    assert len(result.candidates) == 16 and len(result.quality) == 5
    assert sum(c.selected for c in result.candidates) == 1
    assert all(c.active_knee and c.meets_budget and c.eligible for c in result.candidates)
    assert {c.prediction.group for c in result.candidates} == {8, 16}


@pytest.mark.parametrize('flat', [False, True], ids=['no_knee', 'flat_quality'])
def test_normal_miss_uses_median_group_and_preserves_detection_and_queried_trace(flat):
    repo = Repo(scores=(0.7,) * 5 if flat else (0.9, 0.8, 0.7, 0.6, 0.5))
    calls = []
    class EmptyLocator:
        all_knees = []
    def locator(**kwargs):
        calls.append(kwargs)
        return EmptyLocator()
    # Cheaper outside G=32 must not defeat the median restriction. Within G=32,
    # the cheapest frequency misses the budget, and the fastest costs more.
    repo.cell(128, 1122, energy_j_per_token=0.1)
    repo.cell(32, 408, tpot_s=3, energy_j_per_token=0.2)
    repo.cell(32, 612, tpot_s=2, energy_j_per_token=2)
    repo.cell(32, 1122, tpot_s=0.1, energy_j_per_token=3)
    result = choose(repo, KneeAdapter(locator))
    assert pair(result) == (32, 612)
    assert result.selection_path == 'energy_min' and result.predicted_feasible is True
    assert result.selected_quality == 0.7
    assert result.knee == KneeResult('not_found', reason='flat_quality' if flat else 'no_knee')
    assert len(calls) == (0 if flat else 1)
    assert len(result.candidates) == 7 and sum(c.selected for c in result.candidates) == 1
    assert sum(c.active_knee for c in result.candidates) == 7
    assert sum(c.eligible for c in result.candidates) == 7
    for candidate in result.candidates:
        active = candidate.prediction.group == 32
        assert candidate.active_knee is active
        assert candidate.eligible is (active and candidate.meets_budget)
        assert ('knee_excluded' in candidate.reasons) is (not active)
    assert all(c.prediction.frequency_mhz != 408 for c in result.candidates)
    assert repo.query_args == (1024, (32,), 2.0)


def test_detector_recovery_applies_time_condition_to_all_groups():
    repo = Repo()
    repo.cell(128, 1122, energy_j_per_token=0.1)
    result = choose(repo, Adapter(KneeResult('error_recovered', reason='test')))
    assert pair(result) == (128, 1122)
    assert all(c.active_knee for c in result.candidates)
    assert result.knee.status == 'error_recovered'


@pytest.mark.parametrize('reason', ['no_knee', 'flat_quality'])
def test_empty_median_time_set_uses_global_latency_fallback(reason):
    repo = Repo(target=0.5)
    repo.cell(8, 408, tpot_s=0.3, energy_j_per_token=0.1)
    repo.cell(128, 1020, tpot_s=0.2, energy_j_per_token=100)
    result = choose(repo, Adapter(KneeResult('not_found', reason=reason)))
    assert pair(result) == (128, 1020)
    assert result.selection_path == 'latency_fallback' and result.predicted_feasible is True
    assert result.knee == KneeResult('not_found', reason=reason)
    assert len(result.candidates) == 1 and not any(c.eligible for c in result.candidates)
    assert sum(c.active_knee for c in result.candidates) == 0
    selected = next(c for c in result.candidates if c.selected)
    assert selected.active_knee is False and selected.eligible is False
    assert selected.meets_budget is True
    assert selected.reasons == ('knee_excluded', 'latency_fallback')


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
    assert repo.calls == ['quality', 'knee', 'target', 'clock', 'query']
    assert repo.query_args == (1024, GROUPS, 1.0)
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
def test_invalid_or_backward_clock_is_fatal_before_decode_query(now):
    repo = Repo()
    with pytest.raises(ContractError):
        DecodeController(repo, Adapter(), clock=lambda: now).choose(1024, 128, 123, 100)
    assert repo.calls == ['quality', 'target']


@pytest.mark.parametrize('target', [0, -1, float('nan'), float('inf'), True, '1'])
def test_invalid_target_is_fatal_before_clock(target):
    repo = Repo(target=target)
    with pytest.raises(ContractError):
        DecodeController(repo, Adapter(), clock=lambda: pytest.fail('invalid target')).choose(1024, 2, 123, 0)
    assert repo.calls == ['quality', 'target']


@pytest.mark.parametrize('mode', ['missing_quality', 'duplicate_quality', 'wrong_quality_length'])
def test_all_five_quality_rows_are_mandatory(mode):
    repo = Repo()
    if mode == 'missing_quality':
        repo.qualities = repo.qualities[:-1]
    elif mode == 'duplicate_quality':
        repo.qualities = (repo.qualities[0],) * 5
    else:
        repo.qualities = (replace(repo.qualities[0], length=1536, bracket=Bracket(1536, 1536, 0)),) + repo.qualities[1:]
    with pytest.raises(ContractError):
        choose(repo)
    assert repo.calls == ['quality']


@pytest.mark.parametrize('fallback', [False, True])
def test_exact_ties_use_quality_then_energy_or_latency_then_group_frequency(fallback):
    repo = Repo(scores=(0.5, 0.9, 0.9, 0.8, 0.7), target=0.1 if fallback else 2)
    # Every primary metric tied; group16 wins Q, then beats group32 by G, f408 by f.
    assert pair(choose(repo, Adapter(KneeResult('found', GROUPS)))) == (16, 408)
    if fallback:
        repo.cell(32, 510, energy_j_per_token=9)
    else:
        repo.cell(32, 510, tpot_s=0.9)
    assert pair(choose(repo, Adapter(KneeResult('found', GROUPS)))) == (32, 510)


@pytest.mark.parametrize('fallback', [False, True])
def test_single_binary64_step_beats_quality_tiebreak(fallback):
    repo = Repo(target=0.1 if fallback else 2)
    if fallback:
        repo.cell(128, 1122, tpot_s=math.nextafter(1.0, 0))
    else:
        repo.cell(128, 1122, energy_j_per_token=math.nextafter(10.0, 0))
    assert pair(choose(repo, Adapter(KneeResult('found', GROUPS)))) == (128, 1122)


@pytest.mark.parametrize('field,fallback', [('energy_j_per_token', False),
                                         ('tpot_s', False), ('tpot_s', True)])
def test_foreign_integer_metrics_use_canonical_binary64_ties_and_trace(field, fallback):
    base = 2**53
    repo = Repo(target=1 if fallback else 2**54)
    repo.rows = tuple(replace(row, **{field: float(base + 2)}) for row in repo.rows)
    repo.cell(8, 510, **{field: float(base)})
    source = repo.rows[0]
    object.__setattr__(source, field, base + 1)
    object.__setattr__(source.bracket, 'weight', 0)
    # The odd integer rounds to 2**53 in binary64, so 408 wins the final f tie.
    result = choose(repo, Adapter(KneeResult('found', GROUPS)))
    assert pair(result) == (8, 408)
    assert result.selection_path == ('latency_fallback' if fallback else 'energy_min')
    assert getattr(result.selected, field) == float(base)
    assert type(getattr(result.selected, field)) is float
    assert result.selected is not source and result.selected.bracket is not source.bracket
    assert type(result.selected.bracket.weight) is float
    assert all(type(getattr(c.prediction, field)) is float for c in result.candidates)
    assert getattr(source, field) == base + 1 and type(getattr(source, field)) is int
    assert type(source.bracket.weight) is int


def test_canonical_quality_is_used_for_detector_selection_and_trace_without_source_mutation():
    repo = Repo()
    source = repo.qualities[0]
    object.__setattr__(source, 'score', 1)
    object.__setattr__(source.bracket, 'weight', 0)
    class Inspect(Adapter):
        def detect(self, rows):
            assert rows[0] is not source and rows[0].bracket is not source.bracket
            assert type(rows[0].score) is type(rows[0].bracket.weight) is float
            return super().detect(rows)
    result = choose(repo, Inspect(KneeResult('found', GROUPS)))
    assert pair(result) == (8, 408)
    assert result.quality[0] is not source and result.quality[0].bracket is not source.bracket
    assert type(result.selected_quality) is float
    assert all(type(c.quality) is float for c in result.candidates)
    assert type(source.score) is int and type(source.bracket.weight) is int


@pytest.mark.parametrize('table', ['quality', 'query'])
@pytest.mark.parametrize('field,value', [('lower', 1024.0), ('upper', 8193), ('weight', False)])
def test_corrupt_nested_bracket_fails_at_its_lookup_boundary(table, field, value):
    repo = Repo()
    rows = repo.qualities if table == 'quality' else repo.rows
    object.__setattr__(rows[0].bracket, field, value)
    repo.query_result = (repo.rows[0],)
    events = []
    def clock():
        events.append('clock')
        return 100
    with pytest.raises(ContractError):
        DecodeController(repo, Adapter(events=events), clock=clock).choose(1024, 128, 123, 100)
    assert repo.calls == (['quality'] if table == 'quality' else ['quality', 'target', 'query'])
    assert events == ([] if table == 'quality' else ['knee', 'clock'])


def test_budget_boundary_is_exact_and_input_order_does_not_change_choice():
    repo = Repo(target=1)
    repo.cell(16, 510, tpot_s=math.nextafter(1.0, 2), energy_j_per_token=0)
    repo.cell(32, 612, tpot_s=1, energy_j_per_token=1)
    repo.rows = tuple(reversed(repo.rows))
    repo.qualities = tuple(reversed(repo.qualities))
    assert pair(choose(repo, Adapter(KneeResult('found', GROUPS)))) == (32, 612)


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
                (8192, (32, 918), 'energy_min')]
    for length, expected_pair, path in expected:
        result = controller.choose(length, 128, 123, 100)
        assert pair(result) == expected_pair and result.selection_path == path
        assert 1 <= len(result.candidates) <= 8
        assert all(c.eligible for c in result.candidates) if path == 'energy_min' else not any(c.eligible for c in result.candidates)


@pytest.mark.parametrize('bad', [object(), [], {}, (), None])
def test_foreign_or_empty_query_result_is_rejected_after_clock(bad):
    repo = Repo()
    repo.decode_within_budget = lambda *args: bad
    events = []
    with pytest.raises(ContractError):
        DecodeController(repo, Adapter(), clock=lambda: events.append('clock') or 100).choose(
            1024, 128, 123, 100)
    assert events == ['clock']


@pytest.mark.parametrize('mode', ['duplicate', 'too_many', 'foreign_row', 'wrong_length',
                                 'bad_group', 'bad_frequency', 'bad_tpot', 'bad_energy',
                                 'bad_bracket', 'inconsistent_brackets', 'mixed_eligibility',
                                 'unequal_fallback_times'])
def test_corrupt_query_rows_are_rejected_without_silent_filtering(mode):
    repo = Repo()
    row = next(row for row in repo.rows if (row.group, row.frequency_mhz) == (32, 408))
    if mode == 'duplicate':
        rows = (row, row)
    elif mode == 'too_many':
        rows = (row,) * 41
    elif mode == 'foreign_row':
        rows = (object(),)
    elif mode == 'mixed_eligibility':
        rows = (row, repo.rows[0])
    elif mode == 'unequal_fallback_times':
        rows = (replace(repo.rows[0], tpot_s=0.2), replace(repo.rows[1], tpot_s=0.3))
    elif mode == 'inconsistent_brackets':
        rows = (row, replace(repo.rows[1], bracket=Bracket(1024, 2048, 0)))
    else:
        fields = {'wrong_length': ('length', 2048), 'bad_group': ('group', 7),
                  'bad_frequency': ('frequency_mhz', 409), 'bad_tpot': ('tpot_s', math.nan),
                  'bad_energy': ('energy_j_per_token', math.inf), 'bad_bracket': ('bracket', None)}
        field, value = fields[mode]
        object.__setattr__(row, field, value)
        rows = (row,)
    repo.query_result = rows
    with pytest.raises(ContractError):
        choose(repo)
    assert repo.calls == ['quality', 'target', 'query']


@pytest.mark.parametrize('result,expected_groups', [
    (KneeResult('found', (8, 64)), (8, 64)),
    (KneeResult('not_found', reason='no_knee'), (32,)),
    (KneeResult('error_recovered', reason='detector_error'), GROUPS),
])
def test_target_clock_then_conditional_query_receives_active_groups_and_budget(result, expected_groups):
    repo = Repo(target=2)
    def clock():
        repo.calls.append('clock')
        assert repo.calls == ['quality', 'knee', 'target', 'clock']
        return 1_000_000_100
    decision = DecodeController(repo, Adapter(result, repo.calls), clock=clock).choose(1024, 2, 123, 100)
    assert repo.calls == ['quality', 'knee', 'target', 'clock', 'query']
    assert repo.query_args == (1024, expected_groups, 1.0)
    assert decision.elapsed_ns == 1_000_000_000 and decision.budget_s == 1.0
    assert decision.knee == result
