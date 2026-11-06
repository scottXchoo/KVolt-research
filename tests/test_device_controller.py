import json
import hashlib
from pathlib import Path

import pytest

from kvolt_runtime.device import DeviceController, DeviceProfile, HardwareLock
from kvolt_runtime.errors import ContractError
from kvolt_runtime.execution.contracts import ExecutionError, PreparationError


class Clock:
    def __init__(self):
        self.now = 100

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += int(seconds * 1_000_000_000)


class MemoryIO:
    def __init__(self, profile):
        self.profile = profile
        self.values = {
            str(profile.gpu_path / 'governor'): 'nvhost_podgov',
            str(profile.gpu_path / 'min_freq'): '306000000',
            str(profile.gpu_path / 'max_freq'): '1300500000',
            str(profile.gpu_path / 'cur_freq'): '306000000',
            str(profile.gpu_path / 'available_frequencies'): '306000000 408000000 510000000 612000000 714000000 816000000 918000000 1020000000 1122000000 1300500000',
            str(profile.identity_path): 'agx2-test',
            str(profile.boot_id_path): 'boot-one',
            str(profile.emc_rate_path): '3199000000',
            str(profile.emc_lock_path): '1',
            str(profile.emc_cap_path): '3199000000',
            str(profile.emc_max_rate_path): '3199000000',
            str(profile.emc_halt_path): '1',
            str(profile.tpc_path): profile.tpc_expected,
            str(profile.fan_path): '255',
            str(profile.fan_policy_path): 'user_space',
        }
        for policy in profile.cpu_policy_paths:
            for name in ('scaling_min_freq', 'scaling_max_freq'):
                self.values[str(policy / name)] = '2201600'
        self.writes = []
        self.follow = True
        self.fail_on = None

    def read(self, path):
        return self.values[str(path)]

    def write(self, path, value):
        if self.fail_on == path.name:
            raise PermissionError('denied')
        self.writes.append((path.name, str(value)))
        self.values[str(path)] = str(value)
        low = int(self.values[str(self.profile.gpu_path / 'min_freq')])
        high = int(self.values[str(self.profile.gpu_path / 'max_freq')])
        assert low <= high
        if self.follow and low == high:
            self.values[str(self.profile.gpu_path / 'cur_freq')] = str(low)

    def query_maxn(self, command, timeout_ns):
        return {'name': 'MAXN', 'mode_id': 0, 'raw': 'NV Power Mode: MAXN\n0\n'}

    def query_fan_service(self, command, timeout_ns):
        return 'inactive'


def profile_data(tmp_path):
    return {
        'schema_version': 1, 'name': 'test-explicit', 'qualification': 'provisional',
        'evidence': 'unit fixture, not hardware qualification',
        'gpu_path': str(tmp_path / 'gpu'),
        'identity_path': str(tmp_path / 'serial-number'), 'identity_expected': hashlib.sha256(b'agx2-test').hexdigest(),
        'boot_id_path': str(tmp_path / 'boot-id'),
        'cpu_policy_paths': [str(tmp_path / f'policy{x}') for x in (0, 4, 8)],
        'emc_rate_path': str(tmp_path / 'emc/rate'),
        'emc_lock_path': str(tmp_path / 'emc/mrq_rate_locked'),
        'emc_cap_path': str(tmp_path / 'emc/max_rate'),
        'emc_max_rate_path': str(tmp_path / 'emc/debug_max_rate'),
        'emc_halt_path': str(tmp_path / 'emc/bwmgr_halt'),
        'maxn_command': '/usr/sbin/nvpmodel', 'maxn_mode_id': 0,
        'tpc_path': str(tmp_path / 'tpc_pg_mask'), 'tpc_expected': '0',
        'fan_path': str(tmp_path / 'fan/pwm1'),
        'fan_policy_path': str(tmp_path / 'fan/temp_control'),
        'systemctl_command': '/usr/bin/systemctl',
        'timeout_ns': 30_000_000, 'poll_ns': 10_000_000,
        'consecutive_matches': 2, 'query_timeout_ns': 1_000_000_000,
    }


@pytest.fixture
def device(tmp_path):
    profile = DeviceProfile.from_dict(profile_data(tmp_path))
    io, clock = MemoryIO(profile), Clock()
    path = tmp_path / 'hardware.lock'
    path.touch()
    with HardwareLock(path) as lock:
        yield DeviceController(profile, lock=lock, io=io, clock_ns=clock, sleep=clock.sleep), io, clock


@pytest.mark.parametrize('frequency', [408, 510, 612, 714, 816, 918, 1020, 1122])
def test_all_candidates_fresh_exact_and_noop(device, frequency):
    controller, io, clock = device
    receipt = controller.apply(frequency)
    assert receipt['verified'] and receipt['changed']
    assert receipt['observed']['min_hz'] == receipt['observed']['max_hz'] == receipt['observed']['cur_hz'] == frequency * 1_000_000
    assert receipt['end_ns'] - receipt['start_ns'] == 10_000_000
    writes = list(io.writes)
    again = controller.apply(frequency)
    assert again['verified'] and not again['changed'] and again['changes'] == []
    assert io.writes == writes
    assert again['end_ns'] > again['start_ns']


def test_readonly_prepared_checks_all_cpu_and_emc(device):
    controller, io, _ = device
    assert controller.verify_prepared()['verified']
    assert not io.writes
    io.values[str(controller.profile.cpu_policy_paths[2] / 'scaling_max_freq')] = '2000000'
    with pytest.raises(PreparationError) as caught:
        controller.verify_prepared()
    assert not caught.value.receipt['verified']
    assert not io.writes


def test_prepare_common_state_does_not_require_current_equal(device):
    controller, io, _ = device
    io.values[str(controller.profile.gpu_path / 'cur_freq')] = '510000000'
    receipt = controller.prepare_request()
    assert receipt['observed']['min_hz'] == 306000000
    assert receipt['observed']['max_hz'] == 1122000000
    assert receipt['observed']['cur_hz'] == 510000000
    assert {name for name, _ in io.writes} <= {'governor', 'min_freq', 'max_freq'}


def test_fresh_mismatch_prevents_cached_noop_and_timeout_keeps_receipt(device):
    controller, io, _ = device
    controller.apply(408)
    io.values[str(controller.profile.gpu_path / 'cur_freq')] = '306000000'
    io.follow = False
    with pytest.raises(ExecutionError) as caught:
        controller.apply(408)
    receipt = caught.value.receipt
    assert caught.value.code == 'GPU_APPLY_TIMEOUT'
    assert receipt['changed'] and receipt['changes']
    assert receipt['observed']['cur_hz'] == 306000000
    assert receipt['end_ns'] - receipt['start_ns'] == 30_000_000


def test_restore_original_range_not_transient_current(device):
    controller, io, _ = device
    snapshot = controller.capture()
    controller.apply(1122)
    controller.apply(408)
    receipt = controller.restore(snapshot)
    assert receipt['verified']
    assert receipt['observed']['min_hz'] == 306000000
    assert receipt['observed']['max_hz'] == 1300500000
    assert receipt['observed']['cur_hz'] != snapshot['observed']['cur_hz']
    assert ('max_freq', '1300500000') in io.writes


def test_partial_write_error_keeps_successful_changes_and_restores(device):
    controller, io, _ = device
    snapshot = controller.capture()
    io.fail_on = 'max_freq'
    with pytest.raises(ExecutionError) as caught:
        controller.apply(408)
    assert caught.value.receipt['changes']
    assert isinstance(caught.value.__cause__, PermissionError)
    io.fail_on = None
    assert controller.restore(snapshot)['verified']


def test_gpu_identity_or_lock_mismatch_rejected_before_mutation(device):
    controller, io, _ = device
    io.values[str(controller.profile.identity_path)] = 'another-board'
    with pytest.raises(PreparationError):
        controller.apply(408)
    assert not io.writes


@pytest.mark.parametrize('key,value', [('timeout_ns', 0), ('poll_ns', 0), ('consecutive_matches', True), ('query_timeout_ns', 1.5), ('qualification', 'qualified'), ('cpu_policy_paths', ['/only-one'])])
def test_explicit_profiles_reject_invalid_settings(tmp_path, key, value):
    data = profile_data(tmp_path)
    data[key] = value
    with pytest.raises(ContractError):
        DeviceProfile.from_dict(data)


def test_profiles_require_all_fields(tmp_path):
    data = profile_data(tmp_path)
    del data['timeout_ns']
    with pytest.raises(ContractError):
        DeviceProfile.from_dict(data)


def test_lock_reuses_inode_rejects_busy_missing_and_removed(tmp_path):
    path = tmp_path / 'hardware.lock'
    with pytest.raises(PreparationError):
        with HardwareLock(path):
            pass
    path.touch()
    inode = path.stat().st_ino
    with HardwareLock(path) as first:
        assert first.held and path.stat().st_ino == inode
        with pytest.raises(PreparationError) as caught:
            with HardwareLock(path):
                pass
        assert caught.value.code == 'HARDWARE_BUSY'
        path.unlink()
        path.touch()
        with pytest.raises(PreparationError):
            first.assert_held()
    assert not first.held


def test_unknown_frequency_no_mutation(device):
    controller, io, _ = device
    with pytest.raises(ContractError):
        controller.apply(409)
    assert not io.writes
