"""Independent CPU-only H1 edge verification; never touches hardware."""
import fcntl
from pathlib import Path

import pytest

from kvolt_runtime.device import DeviceProfile, HardwareLock
from kvolt_runtime.errors import ContractError
from kvolt_runtime.execution.contracts import ExecutionError, PreparationError
from test_device_controller import device, profile_data


def test_lock_opens_readonly_and_uses_nonblocking_existing_inode(tmp_path, monkeypatch):
    path = tmp_path / 'hardware.lock'
    path.write_bytes(b'preserve experimental lock contents')
    before = path.stat()
    path.chmod(0o444)
    opens, operations = [], []
    original_open, original_flock = Path.open, fcntl.flock

    def observed_open(self, mode='r', *args, **kwargs):
        opens.append(mode)
        return original_open(self, mode, *args, **kwargs)

    def observed_flock(fd, operation):
        operations.append(operation)
        return original_flock(fd, operation)

    monkeypatch.setattr(Path, 'open', observed_open)
    monkeypatch.setattr(fcntl, 'flock', observed_flock)
    with HardwareLock(path) as lock:
        lock.assert_held()
        with pytest.raises(PreparationError) as caught:
            with HardwareLock(path):
                pytest.fail('busy lock entered')
        assert caught.value.code == 'HARDWARE_BUSY'
    assert opens == ['rb', 'rb']
    assert operations == [fcntl.LOCK_EX | fcntl.LOCK_NB,
                          fcntl.LOCK_EX | fcntl.LOCK_NB, fcntl.LOCK_UN]
    assert (path.stat().st_dev, path.stat().st_ino) == (before.st_dev, before.st_ino)
    assert path.read_bytes() == b'preserve experimental lock contents'


def test_lock_reentry_does_not_release_original_owner(tmp_path):
    path = tmp_path / 'hardware.lock'
    path.touch()
    with HardwareLock(path) as lock:
        with pytest.raises(PreparationError) as caught:
            lock.__enter__()
        assert caught.value.code == 'HARDWARE_LOCK_REENTRY'
        lock.assert_held()


def test_all_eight_frequencies_in_both_directions_keep_valid_ranges(device):
    controller, io, _ = device
    candidates = (408, 510, 612, 714, 816, 918, 1020, 1122)
    for frequency in (*candidates, *reversed(candidates)):
        before = controller.capture()['observed']
        io.writes.clear()
        receipt = controller.apply(frequency)
        assert receipt['verified']
        assert receipt['observed']['min_hz'] == receipt['observed']['max_hz'] == frequency * 1_000_000
        if receipt['changed']:
            expected = ['max_freq', 'min_freq'] if frequency * 1_000_000 > before['max_hz'] else ['min_freq', 'max_freq']
            assert [node for node, _ in io.writes] == expected


@pytest.mark.parametrize('node,value', [
    ('governor', 'performance'), ('min_freq', '306000000'),
    ('max_freq', '918000000'), ('cur_freq', '306000000'),
])
def test_noop_requires_fresh_governor_min_max_and_current(device, node, value):
    controller, io, _ = device
    controller.apply(612)
    io.writes.clear()
    io.values[str(controller.profile.gpu_path / node)] = value
    receipt = controller.apply(612)
    assert receipt['changed'] and receipt['verified'] and io.writes
    assert receipt['verification']['samples'] == 2


def test_matching_reads_must_be_consecutive_even_for_noop(device):
    controller, io, clock = device
    controller.apply(612)
    io.writes.clear()
    start = clock.now
    # Initial snapshot matches; subsequent verification resets after a mismatch.
    values = iter(['612000000', '612000000', '306000000', '612000000', '612000000'])
    original_read = io.read

    def read(path):
        return next(values) if path.name == 'cur_freq' else original_read(path)

    io.read = read
    receipt = controller.apply(612)
    assert receipt['verified'] and not receipt['changed']
    assert receipt['verification']['samples'] == 4
    assert receipt['verification']['matched'] == 2
    assert receipt['end_ns'] - start == controller.profile.timeout_ns
    assert not io.writes


def test_nonconsecutive_matches_timeout_without_reusing_previous_confirmation(device):
    controller, io, _ = device
    controller.apply(612)
    values = iter(['612000000', '612000000', '306000000', '612000000', '306000000'])
    original_read = io.read
    io.read = lambda path: next(values) if path.name == 'cur_freq' else original_read(path)
    with pytest.raises(ExecutionError) as caught:
        controller.apply(612)
    receipt = caught.value.receipt
    assert caught.value.code == 'GPU_APPLY_TIMEOUT'
    assert not receipt['verified'] and receipt['verification']['matched'] == 0
    assert receipt['verification']['samples'] == 4


def test_failed_write_receipt_retains_attempt_successes_and_cause(device):
    controller, io, _ = device
    io.fail_on = 'max_freq'
    with pytest.raises(ExecutionError) as caught:
        controller.apply(408)
    receipt = caught.value.receipt
    assert receipt['changed'] and not receipt['verified']
    assert [attempt['succeeded'] for attempt in receipt['write_attempts']] == [True, False]
    assert [change['node'] for change in receipt['changes']] == ['min_freq']
    assert receipt['end_ns'] >= receipt['start_ns']
    assert isinstance(caught.value.__cause__, PermissionError)


@pytest.mark.parametrize('field,value', [
    ('emc_rate_path', '3000000000'), ('emc_cap_path', '3000000000'),
    ('emc_max_rate_path', '3000000000'), ('emc_lock_path', '0'),
    ('emc_halt_path', '0'), ('fan_path', '200'),
    ('fan_policy_path', 'automatic'), ('tpc_path', '1'),
])
def test_unprepared_non_gpu_state_rejected_without_mutation(device, field, value):
    controller, io, _ = device
    io.values[str(getattr(controller.profile, field))] = value
    with pytest.raises(PreparationError) as caught:
        controller.verify_prepared()
    assert caught.value.code == 'DEVICE_NOT_PREPARED'
    assert not io.writes and not caught.value.receipt['changed']


def test_write_that_mutates_then_raises_reports_uncertainty(device):
    controller, io, _ = device
    original_write = io.write

    def write(path, value):
        original_write(path, value)
        raise OSError('failure after partial mutation')

    io.write = write
    with pytest.raises(ExecutionError) as caught:
        controller.apply(408)
    receipt = caught.value.receipt
    assert receipt['mutation_uncertain'] and not receipt['verified']
    assert io.values[str(controller.profile.gpu_path / 'min_freq')] == '408000000'
    assert receipt['write_attempts'][0]['succeeded'] is False
    assert receipt['write_attempts'][0]['end_ns'] >= receipt['write_attempts'][0]['start_ns']


def test_restore_arbitrary_governor_range_ignores_old_current(device):
    controller, io, _ = device
    root = controller.profile.gpu_path
    for node, value in [('governor', 'userspace'), ('min_freq', '612000000'),
                        ('max_freq', '918000000'), ('cur_freq', '714000000')]:
        io.values[str(root / node)] = value
    snapshot = controller.capture()
    controller.apply(408)
    io.writes.clear()
    receipt = controller.restore(snapshot)
    assert receipt['verified']
    assert receipt['observed'] == {'governor': 'userspace', 'min_hz': 612000000,
                                   'max_hz': 918000000, 'cur_hz': 408000000}
    assert [node for node, _ in io.writes] == ['max_freq', 'min_freq', 'governor']


def test_restore_changed_boot_rejected_before_any_write(device):
    controller, io, _ = device
    snapshot = controller.capture()
    controller.apply(408)
    io.writes.clear()
    io.values[str(controller.profile.boot_id_path)] = 'boot-two'
    with pytest.raises(ExecutionError) as caught:
        controller.restore(snapshot)
    assert caught.value.code == 'GPU_RESTORE_IDENTITY_MISMATCH'
    assert not io.writes and not caught.value.receipt['changed']


def test_lock_inode_replacement_blocks_apply_before_any_io_write(device):
    controller, io, _ = device
    controller.lock.path.unlink()
    controller.lock.path.touch()
    with pytest.raises(PreparationError) as caught:
        controller.apply(408)
    assert caught.value.code == 'HARDWARE_LOCK_REPLACED'
    assert not io.writes


@pytest.mark.parametrize('key', ['timeout_ns', 'poll_ns', 'query_timeout_ns'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, 86_400_000_000_001])
def test_profile_requires_finite_explicit_integer_timeouts(tmp_path, key, value):
    data = profile_data(tmp_path)
    data[key] = value
    with pytest.raises(ContractError):
        DeviceProfile.from_dict(data)


@pytest.mark.parametrize('key', ['timeout_ns', 'poll_ns', 'query_timeout_ns', 'consecutive_matches'])
def test_profile_does_not_supply_missing_timing_defaults(tmp_path, key):
    data = profile_data(tmp_path)
    del data[key]
    with pytest.raises(ContractError):
        DeviceProfile.from_dict(data)


def test_profile_rejects_unachievable_consecutive_match_count(tmp_path):
    data = profile_data(tmp_path)
    data['consecutive_matches'] = 5
    with pytest.raises(ContractError):
        DeviceProfile.from_dict(data)
