"""Only GPU governor/min/max are writable. All observations are fresh."""
from pathlib import Path
import hashlib
import re
import subprocess
import time

from ..errors import ContractError
from ..execution.contracts import ExecutionError, PreparationError
from .profile import DeviceProfile

FREQUENCIES_MHZ = (408, 510, 612, 714, 816, 918, 1020, 1122)


class SysfsIO:
    def read(self, path):
        # Device-tree identity files may have a terminating NUL.
        return Path(path).read_text(encoding='utf-8').rstrip('\x00\n').strip()

    def write(self, path, value):
        Path(path).write_text(str(value), encoding='ascii')

    def query_maxn(self, command, timeout_ns):
        result = subprocess.run([str(command), '-q'], check=True, capture_output=True,
                                text=True, timeout=timeout_ns / 1_000_000_000)
        match = re.fullmatch(r'NV Power Mode:\s*([^\n]+)\n\s*(\d+)\s*', result.stdout.strip())
        if not match:
            raise ValueError('unexpected nvpmodel -q response')
        return {'name': match.group(1).strip(), 'mode_id': int(match.group(2)), 'raw': result.stdout}

    def query_fan_service(self, command, timeout_ns):
        result = subprocess.run([str(command), 'is-active', 'nvfancontrol'], check=False,
                                capture_output=True, text=True, timeout=timeout_ns / 1_000_000_000)
        state = result.stdout.strip()
        # systemctl returns 3 for a known inactive unit; unknown/failed is not prepared.
        if (state, result.returncode) not in (('active', 0), ('inactive', 3)):
            raise ValueError('unexpected nvfancontrol service query response')
        return state


class DeviceController:
    def __init__(self, profile, *, lock, io=None, clock_ns=time.monotonic_ns, sleep=time.sleep):
        if not isinstance(profile, DeviceProfile):
            raise ContractError('device requires a validated DeviceProfile')
        self.profile, self.lock = profile, lock
        self.io, self.clock_ns, self.sleep = io or SysfsIO(), clock_ns, sleep

    def _read_int(self, path):
        raw = self.io.read(path)
        if not re.fullmatch(r'\d+', raw):
            raise ValueError(f'invalid integer readback at {path}')
        return int(raw)

    def _identity(self):
        serial = self.io.read(self.profile.identity_path)
        identity = hashlib.sha256(serial.encode('utf-8')).hexdigest()
        boot = self.io.read(self.profile.boot_id_path)
        if not serial or identity != self.profile.identity_expected or not boot:
            raise PreparationError('DEVICE_IDENTITY_MISMATCH', 'device', 'physical identity mismatch or missing boot ID')
        return {'physical_id_sha256': identity, 'boot_id': boot}

    def _gpu(self):
        root = self.profile.gpu_path
        result = {'governor': self.io.read(root / 'governor'),
                  'min_hz': self._read_int(root / 'min_freq'),
                  'max_hz': self._read_int(root / 'max_freq'),
                  'cur_hz': self._read_int(root / 'cur_freq')}
        if not result['governor'] or result['min_hz'] <= 0 or result['max_hz'] < result['min_hz']:
            raise ValueError('invalid GPU settings readback')
        return result

    def _receipt(self, operation):
        return {'operation': operation, 'start_ns': self.clock_ns(), 'end_ns': None,
                'changed': False, 'changes': [], 'write_attempts': [], 'observed': None,
                'verified': False, 'profile': self.profile.as_dict(),
                'verification': {'timeout_ns': self.profile.timeout_ns, 'poll_ns': self.profile.poll_ns,
                                 'consecutive_matches': self.profile.consecutive_matches, 'samples': 0}}

    def _run(self, operation, work):
        receipt = self._receipt(operation)
        try:
            self.lock.assert_held()
            receipt['identity'] = self._identity()
            result = work(receipt)
            receipt['end_ns'] = self.clock_ns()
            return result
        except Exception as error:
            receipt['end_ns'] = self.clock_ns()
            if isinstance(error, ExecutionError):
                error.receipt = receipt
                raise
            error_type = PreparationError if operation in ('capture', 'verify_prepared', 'prepare_request') else ExecutionError
            raise error_type(f'GPU_{operation.upper()}_FAILED', 'device', str(error), receipt=receipt) from error

    def capture(self):
        def work(receipt):
            receipt['observed'] = self._gpu()
            receipt['verified'] = True
            return receipt
        return self._run('capture', work)

    def verify_prepared(self):
        def work(receipt):
            p = self.profile
            observed = {'cpu_policies': []}
            receipt['observed'] = observed
            for path in p.cpu_policy_paths:
                observed['cpu_policies'].append({'path': str(path),
                    'min_khz': self._read_int(path / 'scaling_min_freq'),
                    'max_khz': self._read_int(path / 'scaling_max_freq')})
            observed.update(emc_rate_hz=self._read_int(p.emc_rate_path),
                            emc_locked=self._read_int(p.emc_lock_path),
                            emc_cap_hz=self._read_int(p.emc_cap_path),
                            emc_max_rate_hz=self._read_int(p.emc_max_rate_path),
                            emc_halted=self._read_int(p.emc_halt_path),
                            maxn=self.io.query_maxn(p.maxn_command, p.query_timeout_ns),
                            tpc=self.io.read(p.tpc_path), fan_pwm=self._read_int(p.fan_path),
                            fan_policy=self.io.read(p.fan_policy_path),
                            fan_service=self.io.query_fan_service(p.systemctl_command, p.query_timeout_ns))
            valid = (all(policy['min_khz'] == policy['max_khz'] == 2201600 for policy in observed['cpu_policies'])
                     and observed['emc_rate_hz'] == observed['emc_cap_hz'] == observed['emc_max_rate_hz'] == 3199000000
                     and observed['emc_locked'] == observed['emc_halted'] == 1 and observed['fan_pwm'] == 255
                     and observed['fan_policy'] == 'user_space' and observed['fan_service'] == 'inactive'
                     and observed['maxn']['name'] == 'MAXN' and observed['maxn']['mode_id'] == p.maxn_mode_id
                     and observed['tpc'] == p.tpc_expected)
            if not valid:
                raise PreparationError('DEVICE_NOT_PREPARED', 'device', 'CPU/EMC/MAXN/TPC/fan preparation mismatch')
            receipt['verified'] = True
            return receipt
        return self._run('verify_prepared', work)

    def _write(self, name, value, receipt):
        self.lock.assert_held()
        attempt = {'node': name, 'value': str(value), 'start_ns': self.clock_ns(), 'succeeded': False}
        receipt['write_attempts'].append(attempt)
        try:
            self.io.write(self.profile.gpu_path / name, value)
        except Exception as error:
            attempt.update(end_ns=self.clock_ns(), error={'type': type(error).__name__, 'message': str(error)})
            receipt['mutation_uncertain'] = True
            raise
        attempt.update(succeeded=True, end_ns=self.clock_ns())
        receipt['changes'].append(dict(attempt))
        receipt['changed'] = True

    def _set_range(self, governor, minimum, maximum, before, receipt):
        if before['governor'] != governor:
            self._write('governor', governor, receipt)
        # Expanding the opposite bound first keeps min <= max on both directions.
        nodes = [('max_freq', maximum), ('min_freq', minimum)] if minimum > before['max_hz'] else [('min_freq', minimum), ('max_freq', maximum)]
        for name, value in nodes:
            self._write(name, value, receipt)

    def _verify(self, governor, minimum, maximum, current, receipt):
        deadline = receipt['start_ns'] + self.profile.timeout_ns
        consecutive = 0
        while True:
            self.lock.assert_held()
            observed = self._gpu()
            receipt['observed'] = observed
            receipt['verification']['samples'] += 1
            now = self.clock_ns()
            matches = (observed['governor'] == governor and observed['min_hz'] == minimum
                       and observed['max_hz'] == maximum and (current is None or observed['cur_hz'] == current))
            consecutive = consecutive + 1 if matches else 0
            receipt['verification']['matched'] = consecutive
            if consecutive >= self.profile.consecutive_matches and now <= deadline:
                receipt['verified'] = True
                return receipt
            if now >= deadline:
                raise ExecutionError(f'GPU_{receipt["operation"].upper()}_TIMEOUT', 'device', 'fresh GPU readback verification deadline expired')
            self.sleep(min(self.profile.poll_ns, deadline - now) / 1_000_000_000)

    def prepare_request(self):
        def work(receipt):
            before = self._gpu()
            receipt['before'] = before
            if not (before['governor'] == 'nvhost_podgov' and before['min_hz'] == 306000000 and before['max_hz'] == 1122000000):
                self._set_range('nvhost_podgov', 306000000, 1122000000, before, receipt)
            return self._verify('nvhost_podgov', 306000000, 1122000000, None, receipt)
        return self._run('prepare_request', work)

    def apply(self, frequency_mhz):
        if type(frequency_mhz) is not int or frequency_mhz not in FREQUENCIES_MHZ:
            raise ContractError('GPU frequency must be one of the eight Method candidates in integer MHz')
        target = frequency_mhz * 1_000_000
        def work(receipt):
            receipt['target_hz'] = target
            available = [int(value) for value in self.io.read(self.profile.gpu_path / 'available_frequencies').split()]
            if target not in available:
                raise ExecutionError('GPU_FREQUENCY_UNAVAILABLE', 'device', 'selected GPU frequency is not available')
            before = self._gpu()
            receipt['before'] = before
            if not (before['governor'] == 'nvhost_podgov' and before['min_hz'] == before['max_hz'] == before['cur_hz'] == target):
                self._set_range('nvhost_podgov', target, target, before, receipt)
            return self._verify('nvhost_podgov', target, target, target, receipt)
        return self._run('apply', work)

    def restore(self, snapshot):
        if type(snapshot) is not dict or snapshot.get('operation') != 'capture' or not snapshot.get('verified'):
            raise ContractError('restoration requires a successful capture receipt')
        original = snapshot.get('observed')
        if (type(original) is not dict or type(original.get('governor')) is not str or not original['governor']
                or type(original.get('min_hz')) is not int or type(original.get('max_hz')) is not int
                or not 0 < original['min_hz'] <= original['max_hz']):
            raise ContractError('invalid captured GPU settings')
        def work(receipt):
            if snapshot.get('profile') != self.profile.as_dict() or snapshot.get('identity') != receipt['identity']:
                raise ExecutionError('GPU_RESTORE_IDENTITY_MISMATCH', 'device', 'snapshot profile, physical device or boot differs')
            before = self._gpu()
            receipt['before'] = before
            if any(before[name] != original[name] for name in ('governor', 'min_hz', 'max_hz')):
                # Restore bounds under the current governor, then original governor.
                self._set_range(before['governor'], original['min_hz'], original['max_hz'], before, receipt)
                if before['governor'] != original['governor']:
                    self._write('governor', original['governor'], receipt)
            return self._verify(original['governor'], original['min_hz'], original['max_hz'], None, receipt)
        return self._run('restore', work)
