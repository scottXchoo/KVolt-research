"""Independent TPOT allowance boundaries and policy/artifact consistency."""
from copy import deepcopy
from dataclasses import asdict, replace
import math

import pytest

from kvolt_runtime import config
from kvolt_runtime.errors import ContractError
from kvolt_runtime.policy.decode import DecodeController
from kvolt_runtime.policy.mock import MockLUTRepository
from kvolt_runtime.policy.tpot import tpot_policy
from kvolt_runtime.policy.types import KneeResult, PERFORMANCE_LENGTHS
from kvolt_runtime.qualification.audit import audit_policy, audit_tpot_target
from test_policy_decode import Repo, Adapter


@pytest.mark.parametrize('ratio,expected', [(0, 1.0), (.1, 1.1), (.2, 1.2), (.3, 1.3)])
def test_allowance_changes_only_decode_threshold(ratio, expected):
    repo = Repo(target=1)
    decision = DecodeController(repo, Adapter(), clock=lambda: 100,
                                tpot_buffer_ratio=ratio).choose(1024, 2, 17, 100)
    assert decision.reference_target_s == repo.decode_target(1024) == 1
    assert decision.tpot_buffer_ratio == ratio
    assert decision.target_s == decision.budget_s == expected


def test_default_snapshot_survives_later_global_edit(monkeypatch):
    monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', .2)
    repo = Repo(target=1)
    controller = DecodeController(repo, Adapter(), clock=lambda: 100)
    monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', .3)
    assert controller.choose(1024, 2, 17, 100).target_s == 1.2
    assert DecodeController(repo, Adapter(), clock=lambda: 100).choose(1024, 2, 17, 100).target_s == 1.3
    with pytest.raises(AttributeError):
        controller.tpot_buffer_ratio = .8


def test_inclusive_effective_threshold_and_nextabove():
    repo = Repo(target=1)
    repo.rows = tuple(replace(row, tpot_s=2, energy_j_per_token=10) for row in repo.rows)
    repo.cell(32, 408, tpot_s=1.2, energy_j_per_token=2)
    repo.cell(32, 510, tpot_s=math.nextafter(1.2, math.inf), energy_j_per_token=1)
    buffered = DecodeController(repo, Adapter(), clock=lambda: 100,
                               tpot_buffer_ratio=.2).choose(1024, 2, 17, 100)
    strict = DecodeController(repo, Adapter(), clock=lambda: 100,
                             tpot_buffer_ratio=0).choose(1024, 2, 17, 100)
    assert buffered.selection_path == 'energy_min'
    assert buffered.selected.frequency_mhz == 408
    assert len(buffered.candidates) == 1
    assert strict.selection_path == 'latency_fallback'


def test_reference_interpolation_precedes_single_multiplication():
    refs = {length: length / 1024 for length in PERFORMANCE_LENGTHS}
    repo = MockLUTRepository(decode_target=refs)
    decision = DecodeController(repo, Adapter(), clock=lambda: 100,
                                tpot_buffer_ratio=.2).choose(1536, 3, 17, 100)
    assert repo.decode_target(1536) == decision.reference_target_s == 1.5
    assert decision.target_s == 1.5 * 1.2
    assert decision.target_s != 1.5 * 1.2 * 1.2


@pytest.mark.parametrize('value', [True, False, None, '0.2', -1, math.nan, math.inf, -math.inf, 10**1000])
def test_invalid_buffer_ratio_fails_before_policy_work(value):
    with pytest.raises(ContractError):
        DecodeController(object(), object(), tpot_buffer_ratio=value)


@pytest.mark.parametrize('value', [True, False, None, '1', 0, -1, math.nan, math.inf, -math.inf, 10**1000])
def test_invalid_raw_target_and_overflow(value):
    with pytest.raises(ContractError):
        config.buffered_tpot_target(value, .2)
    with pytest.raises(ContractError):
        config.buffered_tpot_target(1e308, 1)


@pytest.mark.parametrize('nmax,token', [(1, 17), (512, 128009)])
def test_skip_target_evidence_remains_null(nmax, token):
    decision = DecodeController(object(), object(), clock=lambda: pytest.fail('skip clock')).choose(
        1024, nmax, token, 100)
    assert decision.target_s is decision.reference_target_s is decision.tpot_buffer_ratio is None
    audit_tpot_target(asdict(decision), {'length': 1024}, require_new=True)


def test_audit_retains_legacy_zero_without_reclassifying():
    dc = asdict(DecodeController(Repo(target=1), Adapter(), clock=lambda: 100,
                               tpot_buffer_ratio=0).choose(1024, 2, 17, 100))
    dc.pop('reference_target_s')
    dc.pop('tpot_buffer_ratio')
    audit_tpot_target(dc, {'length': 1024}, Repo(target=1))
    with pytest.raises(ValueError, match='missing TPOT'):
        audit_tpot_target(dc, {'length': 1024}, require_new=True)


@pytest.mark.parametrize('mutation', ['raw', 'ratio', 'effective', 'missing_raw', 'missing_ratio', 'stripped'])
def test_audit_denies_changed_or_stripped_new_decision(mutation):
    dc = asdict(DecodeController(Repo(target=1), Adapter(), clock=lambda: 100).choose(1024, 2, 17, 100))
    if mutation == 'raw': dc['reference_target_s'] = 2
    elif mutation == 'ratio': dc['tpot_buffer_ratio'] = .3
    elif mutation == 'effective': dc['target_s'] = 1
    else:
        if mutation in ('missing_raw', 'stripped'): dc.pop('reference_target_s')
        if mutation in ('missing_ratio', 'stripped'): dc.pop('tpot_buffer_ratio')
    with pytest.raises((ValueError, ContractError)):
        audit_tpot_target(dc, {'length': 1024}, Repo(target=1),
                          expected_policy=tpot_policy(.2), require_new=True)


def test_runner_effective_slo_metadata_and_stable_session(tmp_path, monkeypatch):
    from test_session_runner import setup
    monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', .2)
    runner, prepared, options, backend, _, _, _ = setup(tmp_path, requests=2)
    # Fake measured decode includes 4.5 ms of existing measured phase work per request.
    raw = .004
    runner.repo._decode_target = {length: raw for length in PERFORMANCE_LENGTHS}
    monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', .3)
    session = runner.run(prepared, options)
    assert session['exit_code'] == 0
    assert session['tpot_policy'] == tpot_policy(.2)
    assert session['provenance']['runtime']['tpot_policy'] == tpot_policy(.2)
    for result in session['results']:
        dc = result['decisions']['decode']
        measured = result['timing']['mean_tpot_s']
        assert raw < measured <= raw * 1.2
        assert dc['target_s'] == raw * 1.2
        assert dc['reference_target_s'] == raw
        assert result['slo']['mean_tpot_met'] is True
        assert result['schema_version'] == 'kvolt-request-result-v2'


def test_runner_rejects_runtime_metadata_ratio_mismatch_before_hardware(tmp_path):
    from test_session_runner import setup
    runner, prepared, options, backend, events, _, _ = setup(tmp_path)
    result = runner.run(prepared, options, provenance={'runtime': {'tpot_policy': tpot_policy(.3)}})
    assert result['exit_code'] != 0 and backend.loads == 0
    assert 'capture' not in events


def test_new_schema_requires_policy_even_if_decision_fields_stripped(tmp_path):
    from test_session_runner import setup
    runner, prepared, options, _, _, _, _ = setup(tmp_path)
    result = runner.run(prepared, options)['results'][0]
    result.pop('tpot_policy')
    for name in ('reference_target_s', 'tpot_buffer_ratio'):
        result['decisions']['decode'].pop(name)
    with pytest.raises(ContractError, match='policy identity'):
        audit_policy(result)


def test_runtime_metadata_reports_real_default():
    from kvolt_runtime.cli import _runtime_metadata
    assert config.TPOT_BUFFER_RATIO == .2
    assert _runtime_metadata()['tpot_policy'] == tpot_policy(.2)


@pytest.mark.parametrize('measured,met', [(1.2, True), (math.nextafter(1.2, math.inf), False)])
def test_audit_slo_inclusive_threshold_and_reversed_result(measured, met):
    from kvolt_runtime.qualification.audit import audit_slo
    result = {'timing': {'valid': True, 'ttft_s': .5, 'mean_tpot_s': measured},
              'decisions': {'prefill': {'target_s': 1}, 'decode': {'target_s': 1.2}},
              'slo': {'ttft_met': True, 'mean_tpot_met': met}}
    audit_slo(result)
    result['slo']['mean_tpot_met'] = not met
    with pytest.raises(ValueError, match='recorded SLO'):
        audit_slo(result)


def test_default_audit_report_compares_effective_target_and_preserves_raw():
    from pathlib import Path
    from tools.latest_tpot_audit import build_report
    root = Path(__file__).resolve().parents[1]
    report = build_report(root/'data/luts/agx2-runtime-repeat5',
                          root/'data/luts/agx2-reference-repeat10', run_policy=False)
    assert report['inputs']['reference_lut_dir'] == str((root/'data/luts/agx2-reference-repeat10').resolve())
    assert report['context']['tpot_buffer_ratio'] == .2
    assert [row['all40']['time_feasible_count'] for row in report['lengths']] == [20, 20, 20, 20, 20, 20, 21, 22]
    for row in report['lengths']:
        assert row['target_s'] == row['reference_target_s'] * 1.2
        assert row['tpot_buffer_ratio'] == .2


# A complete independent saved power/session fixture proves metadata validation
# without substituting the audit implementation or claiming GPU execution.
from test_qualification_audit import session_bundle


@pytest.mark.parametrize('mutation', [None, 'session_missing', 'runtime_missing', 'result_missing',
    'session_ratio', 'runtime_ratio', 'result_ratio', 'result_schema', 'slo'])
def test_saved_v2_session_policy_tamper_is_rejected(session_bundle, mutation):
    from kvolt_runtime.qualification.audit import audit_session
    root, session, result, save = session_bundle
    policy = tpot_policy(.2)
    session.update(schema_version='kvolt-session-v2', tpot_policy=deepcopy(policy))
    session['provenance']['runtime']['tpot_policy'] = deepcopy(policy)
    result.update(schema_version='kvolt-request-result-v2', tpot_policy=deepcopy(policy))
    result['decisions']['decode'].update(reference_target_s=None, tpot_buffer_ratio=None,
                                        target_s=None, budget_s=None)
    result['slo']['ttft_met'] = result['timing']['ttft_s'] <= result['decisions']['prefill']['target_s']
    if mutation == 'session_missing': session.pop('tpot_policy')
    elif mutation == 'runtime_missing': session['provenance']['runtime'].pop('tpot_policy')
    elif mutation == 'result_missing': result.pop('tpot_policy')
    elif mutation == 'session_ratio': session['tpot_policy']['ratio'] = .3
    elif mutation == 'runtime_ratio': session['provenance']['runtime']['tpot_policy']['ratio'] = .3
    elif mutation == 'result_ratio': result['tpot_policy']['ratio'] = .3
    elif mutation == 'result_schema': result['schema_version'] = 'kvolt-request-result-v1'
    elif mutation == 'slo': result['slo']['ttft_met'] = not result['slo']['ttft_met']
    save()
    if mutation is None:
        assert audit_session(root/'session.json')['status'] == 'passed'
    else:
        with pytest.raises((ValueError, ContractError)):
            audit_session(root/'session.json')


@pytest.mark.parametrize('runtime', [None, {}, {'tpot_policy': None}])
def test_runner_rejects_missing_runtime_policy_before_hardware(tmp_path, runtime):
    from test_session_runner import setup
    runner, prepared, options, backend, events, _, _ = setup(tmp_path)
    session = runner.run(prepared, options, provenance={'runtime': runtime})
    assert session['exit_code'] != 0 and backend.loads == 0
    assert 'capture' not in events
    assert session['results'][0]['tpot_policy'] == session['tpot_policy']


@pytest.mark.parametrize('cause', ['stop', 'device', 'backend', 'request'])
def test_not_run_v2_results_retain_session_policy(tmp_path, cause):
    from test_session_runner import setup
    kwargs = {'requests': 2}
    if cause == 'device': kwargs['device_fail'] = 'verify'
    elif cause == 'backend': kwargs['fail'] = 'load'
    elif cause == 'request': kwargs['fail'] = 'prefill'
    runner, prepared, options, _, _, _, _ = setup(tmp_path, **kwargs)
    if cause == 'stop': runner.stop.request(2)
    session = runner.run(prepared, options)
    assert any(row['generation']['status'] == 'not_run' for row in session['results'])
    assert all(row['schema_version'] == 'kvolt-request-result-v2'
               and row['tpot_policy'] == session['tpot_policy'] for row in session['results'])


def test_v2_skip_cannot_strip_target_fields():
    dc = asdict(DecodeController(object(), object()).choose(1024, 1, 17, 100))
    dc.pop('reference_target_s')
    with pytest.raises(ValueError, match='must remain null'):
        audit_tpot_target(dc, {'length': 1024}, require_new=True)


def test_audit_rejects_boolean_effective_target():
    dc = {'status': 'selected', 'reference_target_s': 1.0, 'tpot_buffer_ratio': 0.0, 'target_s': True}
    with pytest.raises(ValueError, match='effective target'):
        audit_tpot_target(dc, {'length': 1024}, require_new=True)


def test_online_identity_rejects_changed_tpot_policy_before_hardware(tmp_path, monkeypatch):
    import json
    from kvolt_runtime.run_cli import main
    from test_run_cli import arguments, factories
    events, seen = [], []
    seam, _ = factories(events)
    def decode(repo, *args):
        seen.append(repo)
        return object()
    seam = replace(seam, decode_controller=decode)
    assert main(arguments(tmp_path), factories=seam, emit=lambda _: None) == 0
    state = tmp_path/'shared-online.json'
    state.write_text(json.dumps(seen[0].export_state()))
    monkeypatch.setattr(config, 'TPOT_BUFFER_RATIO', .3)
    second = tmp_path/'second'
    second.mkdir()
    events.clear()
    assert main(arguments(second, '--online-lut-state', str(state)), factories=seam, emit=lambda _: None) == 2
    assert 'lock' not in events and 'backend' not in events
    session = json.loads((second/'out/session.json').read_text())
    assert session['errors'][0]['stage'] == 'online_lut'
