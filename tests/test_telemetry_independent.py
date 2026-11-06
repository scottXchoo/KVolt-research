"""Independent analytical and real-spawn CPU verification for M3 T1."""
import ast
import hashlib
import json
import math
import multiprocessing
import os
import pickle
from pathlib import Path
import sys
import time

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.telemetry import (
    INAPowerCollector, PowerProfile, RAILS, RailPaths,
    energy_metrics, read_sample, timing_metrics,
)

S = 10**9


def analytical_rows():
    # Three asynchronous affine curves, including negative signed power.
    curves = ((3, 2, (0, 2, 5)), (-2, 0.5, (0, 3, 6)), (1, -0.25, (0, 2, 6)))
    return [dict(rail=rail, timestamp_ns=t*S, power_w=a+b*t)
            for rail, (a, b, times) in zip(RAILS, curves) for t in times]


def evaluate(rows, *, n=4, completed=True, gap=4*S):
    return energy_metrics(rows, S, 2*S, 4*S if n > 1 else 2*S, n,
                          completed=completed, max_gap_ns=gap)


def test_independent_affine_antiderivatives():
    result = evaluate(analytical_rows())
    assert result['valid'], result
    # Integral of a+bt is a*(v-u)+b*(v²-u²)/2, calculated by hand.
    expected = ((6, 18, 24), (-1.25, -1, -2.25), (0.625, 0.5, 1.125))
    for rail, (pre, dec, total) in zip(RAILS, expected):
        assert result['rails'][rail]['prefill_energy_j'] == pytest.approx(pre)
        assert result['rails'][rail]['decode_energy_j'] == pytest.approx(dec)
        assert result['rails'][rail]['total_energy_j'] == pytest.approx(total)
    assert result['prefill_energy_j'] == pytest.approx(5.375)
    assert result['decode_energy_j'] == pytest.approx(17.5)
    assert result['total_energy_j'] == pytest.approx(22.875)
    assert result['decode_j_per_token'] == pytest.approx(17.5/3)


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf, None, True])
def test_independent_invalid_values_never_filled(bad):
    rows = analytical_rows()
    rows[1]['power_w'] = bad
    result = evaluate(rows)
    assert not result['valid'] and 'invalid_sample' in result['reason']
    assert result['total_energy_j'] is None


@pytest.mark.parametrize('mode,reason', [
    ('missing', 'missing_left_bracket'), ('left_equal', 'missing_left_bracket'),
    ('right_equal', 'missing_right_bracket'), ('reverse', 'nonincreasing_sample_time'),
    ('duplicate', 'nonincreasing_sample_time'), ('error', 'invalid_sample'),
    ('window', 'invalid_read_window'), ('unknown', 'unknown_or_missing_rail'),
])
def test_independent_trace_rejections(mode, reason):
    rows = analytical_rows()
    if mode == 'missing':
        rows = [row for row in rows if row['rail'] != RAILS[1]]
    elif mode == 'left_equal':
        rows[0]['timestamp_ns'] = S
    elif mode == 'right_equal':
        rows[2]['timestamp_ns'] = 4*S
    elif mode == 'reverse':
        rows[0], rows[1] = rows[1], rows[0]
    elif mode == 'duplicate':
        rows[1]['timestamp_ns'] = rows[0]['timestamp_ns']
    elif mode == 'error':
        rows[1]['error'] = {'type': 'OSError'}
    elif mode == 'window':
        rows[1].update(read_start_ns=2*S, read_end_ns=2*S+2)
    else:
        rows[0]['rail'] = 'OTHER'
    result = evaluate(rows)
    assert not result['valid'] and reason in result['reason']
    assert all(result[key] is None for key in
               ('prefill_energy_j', 'decode_energy_j', 'total_energy_j', 'decode_j_per_token'))


def test_independent_gap_single_incomplete_and_disabled():
    assert evaluate(analytical_rows(), gap=4*S)['valid']
    assert 'sample_gap_exceeded' in evaluate(analytical_rows(), gap=4*S-1)['reason']
    one = evaluate(analytical_rows(), n=1)
    assert one['valid'] and one['decode_reason'] == 'no_decode_tokens'
    assert one['decode_energy_j'] is None and one['decode_j_per_token'] is None
    assert one['prefill_energy_j'] == one['total_energy_j'] == pytest.approx(5.375)
    partial = evaluate(analytical_rows(), completed=False)
    assert partial['reason'] == 'incomplete_generation'
    assert partial['total_energy_j'] is None
    assert partial['prefix']['total_energy_j'] == pytest.approx(22.875)
    assert evaluate(None)['reason'] == 'not_collected'
    assert evaluate([])['reason'] == 'empty_trace'
    assert evaluate(analytical_rows(), n=0)['reason'] == 'invalid_boundaries'


def test_independent_timing_precision_and_rejections():
    base = 10**18
    result = timing_metrics(base, [base+1, base+11, base+41], completed=True)
    assert result['ttft_s'] == 1e-9
    assert result['mean_tpot_s'] == 20e-9
    assert result['decode_duration_s'] == 40e-9
    one = timing_metrics(base, [base+1], completed=True)
    assert one['mean_tpot_s'] is None and one['decode_duration_s'] is None
    partial = timing_metrics(base, [base+1, base+11], completed=False)
    assert partial['ttft_s'] is None and partial['mean_tpot_s'] is None
    assert partial['prefix']['mean_tpot_s'] == 10e-9
    assert timing_metrics(base, [], completed=True)['reason'] == 'no_output_tokens'
    for times in ([base], [base+2, base+1], [base+1, base+1]):
        assert not timing_metrics(base, times, completed=True)['valid']


@pytest.fixture
def profile(tmp_path):
    rails = {}
    for index, rail in enumerate(RAILS):
        root = tmp_path / rail
        root.mkdir()
        label, current, voltage = (root / name for name in ('label', 'current', 'voltage'))
        label.write_text(rail+'\n')
        current.write_text(str((-100, 200, 300)[index]))
        voltage.write_text('5000\n')
        rails[rail] = RailPaths(str(label), str(current), str(voltage))
    return PowerProfile(rails, interval_ns=5_000_000, max_gap_ns=500_000_000,
                        ready_timeout_ns=3*S, coverage_timeout_ns=100_000_000,
                        join_timeout_ns=S)


def test_independent_signed_sensor_and_read_window(profile):
    times = iter((100, 110))
    row = read_sample(RAILS[0], profile.rails[RAILS[0]], clock=lambda: next(times))
    assert row['error'] is None
    assert row['current_ma'] == -100 and row['voltage_mv'] == 5000
    assert row['power_w'] == -0.5
    assert (row['read_start_ns'], row['timestamp_ns'], row['read_end_ns']) == (100, 105, 110)


def test_independent_profile_json_validation(profile, tmp_path):
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(profile.to_dict()))
    assert PowerProfile.from_json(path).to_dict() == profile.to_dict()
    for key in ('interval_ns', 'max_gap_ns', 'ready_timeout_ns', 'coverage_timeout_ns', 'join_timeout_ns'):
        value = profile.to_dict()
        value[key] = 0
        path.write_text(json.dumps(value))
        with pytest.raises(ContractError):
            PowerProfile.from_json(path)
    path.write_text('{"rails":{},"rails":{}}')
    with pytest.raises(ContractError, match='duplicate'):
        PowerProfile.from_json(path)
    for update in ({'coverage_timeout_ns': True}, {'join_timeout_ns': float('inf')},
                   {'unexpected': 1}, {'rails': {}}):
        path.write_text(json.dumps(dict(profile.to_dict(), **update)))
        with pytest.raises(ContractError):
            PowerProfile.from_json(path)


def test_independent_real_spawn_brackets_and_cleanup(profile):
    before = {p.pid for p in multiprocessing.active_children()}
    collector = INAPowerCollector(profile)
    try:
        ready = collector.start()
        assert ready['ready'], ready
        assert ready['start_method'] == 'spawn'
        assert ready['clock'] == 'time.monotonic_ns'
        t0 = time.monotonic_ns()
        time.sleep(0.015)
        t1 = time.monotonic_ns()
        time.sleep(0.015)
        tn = time.monotonic_ns()
        receipt = collector.finish(tn)
        assert receipt['status'] == 'stopped', receipt['errors']
        assert receipt['errors'] == []
        assert receipt['trace_sha256'] == hashlib.sha256(receipt['raw_trace_jsonl'].encode()).hexdigest()
        for rail in RAILS:
            rows = [row for row in receipt['samples'] if row['rail'] == rail]
            times = [row['timestamp_ns'] for row in rows]
            assert times[0] < t0 and times[-1] > tn
            assert all(b > a for a, b in zip(times, times[1:]))
            assert all(row['timestamp_ns'] == (row['read_start_ns']+row['read_end_ns'])//2 for row in rows)
        energy = energy_metrics(receipt['samples'], t0, t1, tn, 3, completed=True,
                                max_gap_ns=profile.max_gap_ns)
        assert energy['valid'], energy
        assert energy['total_energy_j'] == pytest.approx(2*(tn-t0)/S)
        assert collector.stop() is receipt and collector.finish(tn) is receipt
    finally:
        collector.stop()
    assert {p.pid for p in multiprocessing.active_children()} <= before


def test_independent_prestart_stop_idempotent(profile):
    collector = INAPowerCollector(profile)
    receipt = collector.stop()
    assert receipt['status'] == 'not_started' and receipt['samples'] == []
    assert collector.stop() is receipt
    with pytest.raises(ContractError):
        collector.start()


def test_independent_readiness_failure_is_bounded_and_clean(profile):
    Path(profile.rails[RAILS[0]].label_path).write_text('WRONG\n')
    data = profile.to_dict()
    data['ready_timeout_ns'] = 500_000_000
    collector = INAPowerCollector(PowerProfile(**data))
    started = time.monotonic()
    try:
        ready = collector.start()
        assert not ready['ready']
        receipt = collector.stop()
        codes = {error['code'] for error in receipt['errors']}
        assert 'COLLECTOR_NOT_READY' in codes
        assert 'COLLECTOR_READ_ERROR' in codes
        assert any(row['error'] for row in receipt['samples'])
        assert collector._process is not None and not collector._process.is_alive()
        assert time.monotonic()-started < 4
    finally:
        collector.stop()


def test_independent_right_coverage_timeout(profile):
    collector = INAPowerCollector(profile)
    try:
        assert collector.start()['ready']
        started = time.monotonic()
        receipt = collector.finish(time.monotonic_ns()+10*S)
        assert any(error['code'] == 'COLLECTOR_RIGHT_COVERAGE' for error in receipt['errors'])
        assert not collector._process.is_alive()
        assert time.monotonic()-started < 3
    finally:
        collector.stop()


def failing_clock():
    """Top-level callable remains importable by the spawn child."""
    raise RuntimeError('independent injected worker failure')


def test_independent_worker_exception_preserved(profile):
    collector = INAPowerCollector(profile, clock=failing_clock)
    try:
        assert not collector.start()['ready']
        receipt = collector.stop()
        assert any(error['code'] == 'COLLECTOR_WORKER_ERROR'
                   and error['type'] == 'RuntimeError' for error in receipt['errors'])
        assert not collector._process.is_alive()
    finally:
        collector.stop()


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX FIFO required')
def test_independent_blocked_sensor_forces_bounded_termination(profile):
    # A local FIFO models an indefinitely blocked sensor read, without hardware.
    current = Path(profile.rails[RAILS[0]].current_path)
    current.unlink()
    os.mkfifo(current)
    data = profile.to_dict()
    data.update(ready_timeout_ns=500_000_000, join_timeout_ns=50_000_000)
    collector = INAPowerCollector(PowerProfile(**data))
    started = time.monotonic()
    try:
        assert not collector.start()['ready']
        receipt = collector.stop()
        assert any(error['code'] == 'COLLECTOR_JOIN_TIMEOUT' for error in receipt['errors'])
        assert not collector._process.is_alive()
        assert time.monotonic()-started < 3
    finally:
        collector.stop()


def test_independent_production_imports_are_stdlib_only():
    root = Path(__file__).resolve().parents[1]/'src/kvolt_runtime/telemetry'
    for path in root.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or '').split('.')[0]]
            else:
                continue
            assert set(names) <= sys.stdlib_module_names, (path, names)


def test_independent_profile_deep_mutation_guards(profile):
    source = profile.to_dict()
    original = profile.to_dict()
    frozen = PowerProfile(**source)
    source['rails'][RAILS[0]]['current_path'] = '/changed-source'
    source['rails'].clear()
    source['name'] = 'changed-source'
    assert frozen.to_dict() == original
    with pytest.raises(TypeError):
        frozen.rails[RAILS[0]] = frozen.rails[RAILS[1]]
    with pytest.raises(TypeError):
        del frozen.rails[RAILS[0]]
    with pytest.raises(AttributeError):
        frozen.rails = {}
    for key in ('label_path', 'current_path', 'voltage_path'):
        with pytest.raises(AttributeError):
            setattr(frozen.rails[RAILS[0]], key, '/changed-field')
    exported = frozen.to_dict()
    exported['rails'][RAILS[0]]['voltage_path'] = '/changed-export'
    exported['rails'].pop(RAILS[1])
    assert frozen.to_dict() == original
    restored = pickle.loads(pickle.dumps(frozen))
    assert restored.to_dict() == original
    with pytest.raises(TypeError):
        restored.rails[RAILS[0]] = restored.rails[RAILS[1]]


@pytest.mark.parametrize('field', ['interval_ns', 'max_gap_ns', 'ready_timeout_ns',
                                  'coverage_timeout_ns', 'join_timeout_ns', 'poll_interval_ns'])
def test_independent_all_profile_integer_bounds(profile, field):
    limit = 86_400_000_000_000
    for bad in (0, -1, True, False, limit+1, 10**100, 1.0, math.inf, math.nan):
        value = profile.to_dict()
        value[field] = bad
        with pytest.raises(ContractError):
            PowerProfile(**value)
    for good in (1, limit):
        value = profile.to_dict()
        # Keep the separate max_gap >= interval invariant satisfied.
        value.update(interval_ns=1, max_gap_ns=limit)
        value[field] = good
        assert getattr(PowerProfile(**value), field) == good


@pytest.mark.parametrize('field', ['name', 'evidence', 'label_path', 'current_path', 'voltage_path'])
def test_independent_utf8_rejection_direct_and_json(profile, tmp_path, field):
    path_field = field.endswith('_path')
    for bad in ('', '\ud800', '\udfff', None, 123):
        value = profile.to_dict()
        target = value['rails'][RAILS[0]] if path_field else value
        target[field] = '/'+bad if path_field and isinstance(bad, str) and bad else bad
        with pytest.raises(ContractError):
            PowerProfile(**value)
        path = tmp_path/'surrogate-profile.json'
        path.write_text(json.dumps(value, ensure_ascii=True))
        with pytest.raises(ContractError):
            PowerProfile.from_json(path)
    value = profile.to_dict()
    target = value['rails'][RAILS[0]] if path_field else value
    target[field] = '/한글/센서' if path_field else '한글 센서 프로필'
    assert PowerProfile(**value).to_dict() == value
    if path_field:
        target[field] = '/sensor\x00bad'
        with pytest.raises(ContractError):
            PowerProfile(**value)


def test_independent_pickle_roundtrip_actual_spawn(profile):
    restored = pickle.loads(pickle.dumps(profile))
    collector = INAPowerCollector(restored)
    try:
        assert collector.start()['ready']
        receipt = collector.finish(time.monotonic_ns())
        assert receipt['status'] == 'stopped', receipt['errors']
        assert receipt['profile'] == profile.to_dict()
        assert receipt['start_method'] == 'spawn'
        assert not collector._process.is_alive()
    finally:
        collector.stop()
