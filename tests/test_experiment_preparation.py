from pathlib import Path
from types import SimpleNamespace

import pytest

from kvolt_runtime.qualification.session import ExperimentPreparation


class FakeLock:
    def assert_held(self):
        pass


class Controls:
    def __init__(self, profile):
        p = profile
        self.values = {p.emc_rate_path: '3199000000', p.emc_cap_path: '3199000000',
                       p.emc_max_rate_path: '3199000000', p.emc_lock_path: '0', p.emc_halt_path: '0',
                       p.fan_path: '46', p.fan_policy_path: 'user_space', p.tpc_path: '0'}
        for cpu in p.cpu_policy_paths:
            self.values.update({cpu / 'scaling_min_freq': '729600', cpu / 'scaling_max_freq': '2201600',
                                cpu / 'scaling_governor': 'schedutil'})
        self.service = 'active'
        self.actions = []
        self.failure = None

    def read(self, path):
        return self.values[path]

    def write(self, path, value):
        self.actions.append(('write', path, str(value)))
        if self.failure == path:
            self.failure = None
            raise OSError('injected write failure')
        self.values[path] = str(value)

    def query_maxn(self, command, timeout):
        return {'name': 'MAXN', 'mode_id': 0}

    def query_fan_service(self, command, timeout):
        return self.service

    def fan_service(self, command, action, timeout):
        self.actions.append(('fan_service', action))
        self.service = 'active' if action == 'start' else 'inactive'
        if action == 'start':
            self.values[Path('/fan/pwm')] = '48'  # restored service owns the dynamic value


@pytest.fixture
def preparation():
    p = SimpleNamespace(cpu_policy_paths=tuple(Path('/cpu') / str(x) for x in (0, 4, 8)),
                        emc_rate_path=Path('/emc/rate'), emc_cap_path=Path('/emc/cap'),
                        emc_max_rate_path=Path('/emc/max'), emc_lock_path=Path('/emc/lock'),
                        emc_halt_path=Path('/emc/halt'), fan_path=Path('/fan/pwm'),
                        fan_policy_path=Path('/fan/policy'), tpc_path=Path('/gpu/tpc'),
                        tpc_expected='0', maxn_command=Path('/usr/sbin/nvpmodel'), maxn_mode_id=0,
                        systemctl_command=Path('/bin/systemctl'), query_timeout_ns=1_000_000_000)
    io = Controls(p)
    return ExperimentPreparation(p, lock=FakeLock(), io=io), io


def test_explicit_preparation_restores_original_controls(preparation):
    prep, io = preparation
    before = prep.capture()
    prep.prepare()
    assert io.service == 'inactive'
    assert io.values[prep.profile.fan_path] == '255'
    assert all(io.values[path / 'scaling_min_freq'] == '2201600' for path in prep.profile.cpu_policy_paths)
    restored = prep.restore()
    assert restored['verified'] and io.service == 'active'
    assert all(io.values[Path(row['path']) / 'scaling_min_freq'] == str(row['minimum']) for row in before['cpu'])
    assert io.values[prep.profile.emc_lock_path] == io.values[prep.profile.emc_halt_path] == '0'
    assert not restored['dynamic_emc_rate_restored_exactly']
    assert not restored['dynamic_fan_pwm_restored_exactly']
    assert not any(str(action[1]).startswith('/gpu') for action in io.actions)


def test_partial_prepare_failure_is_restorable(preparation):
    prep, io = preparation
    prep.capture()
    io.failure = prep.profile.emc_halt_path
    with pytest.raises(OSError):
        prep.prepare()
    assert prep.restore()['verified']
    assert io.values[prep.profile.emc_lock_path] == '0'


def test_restore_continues_after_one_cpu_failure(preparation):
    prep, io = preparation
    prep.capture()
    prep.prepare()
    io.failure = prep.profile.cpu_policy_paths[0] / 'scaling_min_freq'
    result = prep.restore()
    assert not result['verified'] and result['errors']
    assert io.values[prep.profile.cpu_policy_paths[1] / 'scaling_min_freq'] == '729600'
    assert io.values[prep.profile.emc_lock_path] == '0'
    assert io.service == 'active'


def test_unprepared_maxn_is_never_changed(preparation):
    prep, io = preparation
    io.query_maxn = lambda *args: {'name': 'OTHER', 'mode_id': 1}
    with pytest.raises(ValueError, match='MAXN'):
        prep.capture()
    assert io.actions == []
    assert prep.restore() == {'verified': True, 'changed': False, 'errors': []}


def test_prepare_requires_capture(preparation):
    prep, io = preparation
    with pytest.raises(ValueError, match='capture'):
        prep.prepare()
    assert io.actions == []


def test_inactive_fan_pwm_restores_exactly(preparation):
    prep, io = preparation
    io.service = 'inactive'
    prep.capture()
    prep.prepare()
    result = prep.restore()
    assert result['verified'] and result['dynamic_fan_pwm_restored_exactly']
    assert io.values[prep.profile.fan_path] == '46'


@pytest.mark.parametrize('field', ['emc_rate_path', 'emc_halt_path', 'emc_lock_path'])
def test_emc_internal_write_failure_still_attempts_remaining_controls(preparation, field):
    prep, io = preparation
    prep.capture()
    prep.prepare()
    io.failure = getattr(prep.profile, field)
    result = prep.restore()
    assert not result['verified'] and result['errors']
    # One-shot failure may be an early re-lock attempt; final unlock/halt
    # restoration must still be attempted even if the old rate write failed.
    assert io.values[prep.profile.emc_lock_path] == '0'
    assert io.values[prep.profile.emc_halt_path] == '0'
    assert io.service == 'active'


@pytest.mark.parametrize('field', ['fan_policy_path', 'fan_path'])
def test_fan_internal_write_failure_still_restarts_original_service(preparation, field):
    prep, io = preparation
    prep.capture()
    prep.prepare()
    io.failure = getattr(prep.profile, field)
    result = prep.restore()
    assert not result['verified'] and result['errors']
    assert io.service == 'active'
    assert io.values[prep.profile.emc_lock_path] == '0'


def test_preparation_receipt_does_not_grow_with_restoration(preparation):
    prep, io = preparation
    prep.capture()
    receipt = prep.prepare()
    count = len(receipt['changes'])
    assert all(row['phase'] == 'prepare' for row in receipt['changes'])
    prep.restore()
    assert len(receipt['changes']) == count
    assert all(row['phase'] == 'prepare' for row in receipt['changes'])
    assert any(row['phase'] == 'restore' for row in prep.changes)
    after = prep.capture()
    assert after['fan_service'] == 'active'
    assert after['emc']['lock'] == after['emc']['halt'] == 0
