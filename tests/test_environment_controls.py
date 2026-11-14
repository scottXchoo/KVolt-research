"""Reversible controls preserve partial failures and original hardware policy."""
import subprocess

import pytest

from experiment_control.controls import (
    AffinityControl, ClockControl, EmcControl, FanControl, FileValueControl,
    PowerModeControl, SwappinessControl,
)
from experiment_control.contracts import ReadinessError, RestorationError


def node(root, name, value):
    path = root / name
    path.write_text(str(value))
    return path


def test_file_value_roundtrip_and_lock_assertion(tmp_path):
    path = node(tmp_path, 'value', 4)
    checks = []
    control = FileValueControl('value', path, 8, assert_held=lambda: checks.append(True))
    old = control.capture()
    control.apply(old)
    assert path.read_text() == '8'
    control.restore(old)
    assert path.read_text() == '4'
    assert len(checks) == 2


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), '', 'a\nb'])
def test_file_value_rejects_ambiguous_input(tmp_path, value):
    with pytest.raises(ValueError):
        FileValueControl('value', tmp_path / 'value', value)


def test_clock_expands_opposite_bound_and_restores_governor(tmp_path, monkeypatch):
    for name, value in [('min_freq', 10), ('max_freq', 20), ('cur_freq', 40), ('governor', 'old')]:
        node(tmp_path, name, value)
    control = ClockControl('gpu', tmp_path, 40, 40, 'new', verify_current=True)
    old = control.capture()
    writes = []
    original = control._write
    def checked_write(path, value):
        writes.append((path.name, str(value)))
        original(path, value)
        assert int((tmp_path / 'min_freq').read_text()) <= int((tmp_path / 'max_freq').read_text())
    monkeypatch.setattr(control, '_write', checked_write)
    control.apply(old)
    assert writes.index(('max_freq', '40')) < writes.index(('min_freq', '40'))
    control.restore(old)
    assert (tmp_path / 'governor').read_text() == 'old'
    assert (tmp_path / 'min_freq').read_text() == '10'
    assert (tmp_path / 'max_freq').read_text() == '20'


def test_clock_partial_apply_can_restore_all_nodes(tmp_path, monkeypatch):
    for name, value in [('min_freq', 10), ('max_freq', 20), ('governor', 'old')]:
        node(tmp_path, name, value)
    control = ClockControl('gpu', tmp_path, 40, 40, 'new')
    old = control.capture()
    original = control._write
    failed = []
    def fail_once(path, value):
        if path.name == 'min_freq' and not failed:
            failed.append(True)
            raise OSError('write interrupted')
        original(path, value)
    monkeypatch.setattr(control, '_write', fail_once)
    with pytest.raises(OSError):
        control.apply(old)
    control.restore(old)
    assert {name: (tmp_path / name).read_text() for name in old} == old


def test_clock_restore_attempts_governor_after_bound_failure(tmp_path, monkeypatch):
    for name, value in [('min_freq', 10), ('max_freq', 20), ('governor', 'old')]:
        node(tmp_path, name, value)
    control = ClockControl('gpu', tmp_path, 15, 20, 'new')
    old = control.capture()
    control.apply(old)
    original = control._write
    def failure(path, value):
        if path.name == 'min_freq':
            raise OSError('cannot restore min')
        original(path, value)
    monkeypatch.setattr(control, '_write', failure)
    with pytest.raises(RestorationError):
        control.restore(old)
    assert (tmp_path / 'governor').read_text() == 'old'


def test_clock_timeout_and_validation(tmp_path):
    for name, value in [('min_freq', 10), ('max_freq', 20), ('cur_freq', 10), ('governor', 'old')]:
        node(tmp_path, name, value)
    control = ClockControl('gpu', tmp_path, 20, 20, verify_current=True,
                           timeout_s=0.001, poll_s=0.001)
    with pytest.raises(ReadinessError):
        control.apply(control.capture())
    for kwargs in ({'minimum': True}, {'maximum': float('nan')}, {'min_file': '../trip_point_0_temp'},
                   {'timeout_s': float('inf')}, {'consecutive': True}, {'poll_s': 0}):
        args = dict(name='gpu', path=tmp_path, minimum=10, maximum=20)
        args.update(kwargs)
        with pytest.raises(ValueError):
            ClockControl(**args)


def test_emc_dynamic_restores_controls_but_not_transient_rate(tmp_path):
    rate, lock, halt = [node(tmp_path, n, v) for n, v in [('rate', 100), ('lock', 0), ('halt', 0)]]
    control = EmcControl(rate, lock, halt, 200)
    old = control.capture()
    control.apply(old)
    control.restore(old)
    assert lock.read_text() == halt.read_text() == '0'
    assert rate.read_text() == '100'


def test_emc_restore_continues_after_rate_failure(tmp_path, monkeypatch):
    rate, lock, halt = [node(tmp_path, n, v) for n, v in [('rate', 100), ('lock', 0), ('halt', 0)]]
    control = EmcControl(rate, lock, halt, 200)
    old = control.capture()
    control.apply(old)
    original = control._write
    def failure(path, value):
        if path == rate:
            raise OSError('rate write failed')
        original(path, value)
    monkeypatch.setattr(control, '_write', failure)
    with pytest.raises(RestorationError):
        control.restore(old)
    assert lock.read_text() == halt.read_text() == '0'


class Service:
    def __init__(self, state='active'):
        self.state, self.calls = state, []
    def __call__(self, args, **kwargs):
        self.calls.append(args)
        if args[1] == 'is-active':
            return subprocess.CompletedProcess(args, 0 if self.state == 'active' else 3, self.state + '\n', '')
        self.state = 'active' if args[1] == 'start' else 'inactive'
        return subprocess.CompletedProcess(args, 0, '', '')


def test_fan_roundtrip_preserves_thermal_policy_and_service(tmp_path):
    pwm, policy = node(tmp_path, 'pwm1', 70), node(tmp_path, 'policy', 'step_wise')
    runner = Service()
    control = FanControl(pwm, 255, runner=runner)
    old = control.capture()
    control.apply(old)
    assert runner.state == 'inactive' and pwm.read_text() == '255'
    control.restore(old)
    assert runner.state == 'active' and policy.read_text() == 'step_wise'


def test_fan_partial_pwm_failure_still_restores_service(tmp_path, monkeypatch):
    pwm = node(tmp_path, 'pwm1', 70)
    runner = Service()
    control = FanControl(pwm, 255, runner=runner)
    old = control.capture()
    def failure(*args):
        raise OSError('pwm unwritable')
    monkeypatch.setattr(control, '_write', failure)
    with pytest.raises(OSError):
        control.apply(old)
    with pytest.raises(RestorationError):
        control.restore(old)
    assert runner.state == 'active'


def test_power_mode_query_only():
    calls = []
    def runner(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, 'NV Power Mode: MAXN\n0\n', '')
    control = PowerModeControl(0, 'MAXN', runner=runner)
    old = control.capture()
    control.apply(old)
    control.restore(old)
    assert all(args == ['/usr/sbin/nvpmodel', '-q'] for args in calls)
    with pytest.raises(ReadinessError):
        PowerModeControl(1, runner=runner).apply(old)


def test_affinity_restores_and_verifies():
    state = [{0, 1, 2}]
    control = AffinityControl({0, 1}, get_affinity=lambda pid: state[0].copy(),
                              set_affinity=lambda pid, cpus: state.__setitem__(0, set(cpus)))
    old = control.capture()
    control.apply(old)
    assert state[0] == {0, 1}
    control.restore(old)
    assert state[0] == {0, 1, 2}


@pytest.mark.parametrize('value', [-1, 201, True, '20'])
def test_swappiness_validation(value):
    with pytest.raises(ValueError):
        SwappinessControl(value)


def test_fan_unknown_service_state_rejected(tmp_path):
    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 4, 'unknown\n', '')
    control = FanControl(node(tmp_path, 'pwm1', 50), 200, runner=runner)
    with pytest.raises(ReadinessError):
        control.capture()


def test_fan_stop_failure_does_not_write_pwm(tmp_path):
    pwm = node(tmp_path, 'pwm1', 50)
    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, 'active\n', '')
    with pytest.raises(ReadinessError):
        FanControl(pwm, 200, runner=runner).apply({'pwm': 50, 'service': 'active'})
    assert pwm.read_text() == '50'


def test_emc_restoration_does_not_require_transient_dynamic_rate(tmp_path, monkeypatch):
    rate, lock, halt = [node(tmp_path, n, v) for n, v in [('rate', 100), ('lock', 0), ('halt', 0)]]
    control = EmcControl(rate, lock, halt, 200)
    old = control.capture()
    control.apply(old)
    original = control._write
    def dynamic(path, value):
        original(path, value)
        if path == lock and value == 0:
            rate.write_text('80')
    monkeypatch.setattr(control, '_write', dynamic)
    control.restore(old)
    assert rate.read_text() == '80'


def test_affinity_unavailable_platform():
    with pytest.raises(ReadinessError):
        AffinityControl({0}, get_affinity=None, set_affinity=None).capture()
