"""Explicit reversible controls; no thermal protection or power-mode changes.

Paths are caller-selected. Profile construction must expose only known control
nodes, never thermal trips or shutdown policies. Snapshot before every mutation.
"""
import os
from pathlib import Path
import re
import subprocess
import time

from .contracts import ReadinessError, RestorationError, finite_number


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be an integer in the permitted range')
    return value


def _text(value, name):
    if (type(value) is not str or not value or value != value.strip()
            or any(c in value for c in ('\x00', '\n', '\r'))):
        raise ValueError(f'{name} must be a nonempty single-line string')
    return value


def _path(value):
    if not isinstance(value, (str, Path)):
        raise ValueError('path must be a string or Path')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or any(c in str(path) for c in ('\x00', '\n', '\r')):
        raise ValueError('path must be absolute without parent traversal')
    return path


def _leaf(value):
    _text(value, 'node filename')
    if not re.fullmatch(r'[A-Za-z0-9_]+', value):
        raise ValueError('node filename must be a simple name')
    return value


def _read(path):
    return path.read_text(encoding='ascii').strip()


def _read_integer(path):
    value = _read(path)
    if not re.fullmatch(r'\d+', value):
        raise ReadinessError(f'invalid integer readback: {path}')
    return int(value)


def _positive_time(value, name):
    finite_number(value, name, minimum=0)
    if value == 0:
        raise ValueError(f'{name} must be positive')
    return value


class _Control:
    def __init__(self, name, assert_held=None):
        self.name = _text(name, 'name')
        if assert_held is not None and not callable(assert_held):
            raise ValueError('assert_held must be callable')
        self.assert_held = assert_held

    def _assert(self):
        if self.assert_held is not None:
            self.assert_held()

    def _write(self, path, value):
        self._assert()
        path.write_text(str(value), encoding='ascii')

    def _restore_attempts(self, operations):
        errors = []
        for label, operation in operations:
            try:
                operation()
            except BaseException as error:
                errors.append(f'{label}: {type(error).__name__}: {error}')
        if errors:
            raise RestorationError('; '.join(errors))


class FileValueControl(_Control):
    def __init__(self, name, path, value, *, assert_held=None):
        super().__init__(name, assert_held)
        self.path = _path(path)
        if type(value) is int:
            value = str(value)
        self.value = _text(value, 'value')
        try:
            self.value.encode('ascii')
        except UnicodeError as error:
            raise ValueError('value must be ASCII') from error

    def capture(self):
        return _text(_read(self.path), 'snapshot')

    def apply(self, snapshot):
        self._write(self.path, self.value)
        if _read(self.path) != self.value:
            raise ReadinessError(f'{self.name} readback mismatch')

    def restore(self, snapshot):
        snapshot = _text(snapshot, 'snapshot')
        def restore():
            self._write(self.path, snapshot)
            if _read(self.path) != snapshot:
                raise ReadinessError(f'{self.name} restoration mismatch')
        self._restore_attempts([('value', restore)])


class ClockControl(_Control):
    def __init__(self, name, path, minimum, maximum, governor=None, *,
                 min_file='min_freq', max_file='max_freq', governor_file='governor',
                 current_file='cur_freq', verify_current=False, timeout_s=1.0,
                 poll_s=0.01, consecutive=1, assert_held=None,
                 clock=time.monotonic, sleep=time.sleep):
        super().__init__(name, assert_held)
        self.path = _path(path)
        self.minimum = _integer(minimum, 'minimum', 1)
        self.maximum = _integer(maximum, 'maximum', self.minimum)
        self.governor = _text(governor, 'governor') if governor is not None else None
        self.min_file, self.max_file, self.governor_file, self.current_file = [
            _leaf(v) for v in (min_file, max_file, governor_file, current_file)]
        if len({self.min_file, self.max_file, self.governor_file, self.current_file}) != 4:
            raise ValueError('clock node filenames must be distinct')
        if type(verify_current) is not bool or (verify_current and minimum != maximum):
            raise ValueError('verify_current requires a fixed clock')
        self.verify_current = verify_current
        self.timeout_s = _positive_time(timeout_s, 'timeout_s')
        self.poll_s = _positive_time(poll_s, 'poll_s')
        self.consecutive = _integer(consecutive, 'consecutive', 1)
        if self.poll_s > self.timeout_s or (consecutive - 1) * poll_s > timeout_s:
            raise ValueError('verification samples must fit within timeout')
        if not callable(clock) or not callable(sleep):
            raise ValueError('clock/sleep must be callable')
        self.clock, self.sleep = clock, sleep

    def capture(self):
        minimum, maximum = _read_integer(self.path/self.min_file), _read_integer(self.path/self.max_file)
        if not 0 < minimum <= maximum:
            raise ReadinessError('invalid original clock bounds')
        return {self.min_file: str(minimum), self.max_file: str(maximum),
                self.governor_file: _text(_read(self.path/self.governor_file), 'original governor')}

    def _range_operations(self, minimum, maximum):
        current_max = _read_integer(self.path/self.max_file)
        pairs = [(self.max_file, maximum), (self.min_file, minimum)] if minimum > current_max else [
            (self.min_file, minimum), (self.max_file, maximum)]
        return [(name, lambda name=name, value=value: self._write(self.path/name, value)) for name, value in pairs]

    def _verify(self, minimum, maximum, governor, current=False):
        deadline, matches = self.clock() + self.timeout_s, 0
        while True:
            valid = (_read_integer(self.path/self.min_file) == minimum
                     and _read_integer(self.path/self.max_file) == maximum
                     and _read(self.path/self.governor_file) == governor
                     and (not current or _read_integer(self.path/self.current_file) == minimum))
            matches = matches + 1 if valid else 0
            now = self.clock()
            if matches >= self.consecutive and now <= deadline:
                return
            if now >= deadline:
                raise ReadinessError(f'{self.name} clock confirmation timeout')
            self.sleep(min(self.poll_s, deadline - now))

    def apply(self, snapshot):
        governor = self.governor if self.governor is not None else snapshot[self.governor_file]
        if self.governor is not None:
            self._write(self.path/self.governor_file, governor)
        for _, operation in self._range_operations(self.minimum, self.maximum):
            operation()
        self._verify(self.minimum, self.maximum, governor, self.verify_current)

    def restore(self, snapshot):
        minimum, maximum = int(snapshot[self.min_file]), int(snapshot[self.max_file])
        governor = _text(snapshot[self.governor_file], 'original governor')
        try:
            operations = self._range_operations(minimum, maximum)
        except Exception as error:
            # Even unreadable bounds must not prevent independent restoration.
            def failed(error=error):
                raise error
            operations = [('bounds_read', failed)]
            operations += [(name, lambda name=name, value=value: self._write(self.path/name, value))
                           for name, value in ((self.min_file, minimum), (self.max_file, maximum))]
        operations += [('governor', lambda: self._write(self.path/self.governor_file, governor)),
                       ('readback', lambda: self._verify(minimum, maximum, governor))]
        self._restore_attempts(operations)


class EmcControl(_Control):
    def __init__(self, rate_path, lock_path, halt_path, rate_hz, *, name='emc', assert_held=None):
        super().__init__(name, assert_held)
        self.rate_path, self.lock_path, self.halt_path = map(_path, (rate_path, lock_path, halt_path))
        if len({self.rate_path, self.lock_path, self.halt_path}) != 3:
            raise ValueError('EMC paths must be distinct')
        self.rate_hz = _integer(rate_hz, 'rate_hz', 1)

    def capture(self):
        snapshot = {'rate': _read_integer(self.rate_path), 'lock': _read_integer(self.lock_path),
                    'halt': _read_integer(self.halt_path)}
        if snapshot['lock'] not in (0, 1) or snapshot['halt'] not in (0, 1) or snapshot['rate'] <= 0:
            raise ReadinessError('invalid EMC snapshot')
        return snapshot

    def apply(self, snapshot):
        for path, value in ((self.lock_path, 1), (self.halt_path, 1), (self.rate_path, self.rate_hz)):
            self._write(path, value)
            if _read_integer(path) != value:
                raise ReadinessError(f'{self.name} readback mismatch: {path}')

    def restore(self, snapshot):
        pairs = ((self.lock_path, 1), (self.halt_path, 1), (self.rate_path, snapshot['rate']),
                 (self.halt_path, snapshot['halt']), (self.lock_path, snapshot['lock']))
        operations = [(str(path), lambda path=path, value=value: self._write(path, value)) for path, value in pairs]
        def verify():
            actual = self.capture()
            if (actual['lock'] != snapshot['lock'] or actual['halt'] != snapshot['halt']
                    or (snapshot['lock'] == 1 and actual['rate'] != snapshot['rate'])):
                raise ReadinessError('EMC restoration mismatch')
        self._restore_attempts(operations + [('readback', verify)])


class FanControl(_Control):
    """Control explicit PWM/service only; never write thermal policy or trip nodes."""
    def __init__(self, pwm_path, pwm, service='nvfancontrol', systemctl='/usr/bin/systemctl', *,
                 name='fan', runner=subprocess.run, assert_held=None, timeout_s=5.0):
        super().__init__(name, assert_held)
        self.pwm_path, self.pwm = _path(pwm_path), _integer(pwm, 'pwm', 0, 255)
        if service != 'nvfancontrol':
            raise ValueError('only nvfancontrol service is supported')
        self.service, self.systemctl = service, _path(systemctl)
        if self.systemctl not in (Path('/usr/bin/systemctl'), Path('/bin/systemctl')):
            raise ValueError('use an explicit system systemctl path')
        if not callable(runner):
            raise ValueError('runner must be callable')
        self.runner, self.timeout_s = runner, _positive_time(timeout_s, 'timeout_s')

    def _command(self, action):
        if action != 'is-active':
            self._assert()
        result = self.runner([str(self.systemctl), action, self.service], capture_output=True,
                             text=True, timeout=self.timeout_s, check=False)
        if action != 'is-active' and result.returncode != 0:
            raise ReadinessError(f'fan service {action} failed ({result.returncode})')
        return result

    def _state(self):
        result = self._command('is-active')
        state = result.stdout.strip()
        if (state, result.returncode) not in (('active', 0), ('inactive', 3)):
            raise ReadinessError('unknown fan service state')
        return state

    def capture(self):
        return {'pwm': _integer(_read_integer(self.pwm_path), 'pwm', 0, 255), 'service': self._state()}

    def apply(self, snapshot):
        self._command('stop')
        if self._state() != 'inactive':
            raise ReadinessError('fan service did not stop')
        self._write(self.pwm_path, self.pwm)
        if _read_integer(self.pwm_path) != self.pwm:
            raise ReadinessError('fan PWM readback mismatch')

    def restore(self, snapshot):
        def verify():
            actual = self.capture()
            if actual['service'] != snapshot['service'] or (snapshot['service'] == 'inactive'
                                                          and actual['pwm'] != snapshot['pwm']):
                raise ReadinessError('fan restoration mismatch')
        self._restore_attempts([
            ('pwm', lambda: self._write(self.pwm_path, snapshot['pwm'])),
            ('service', lambda: self._command('start' if snapshot['service'] == 'active' else 'stop')),
            ('readback', verify)])


class PowerModeControl(_Control):
    """Read-only requirement. Never switch nvpmodel, TPC mask, or reboot."""
    def __init__(self, expected_id, expected_name=None, command='/usr/sbin/nvpmodel', *,
                 name='power_mode', runner=subprocess.run, timeout_s=5.0, assert_held=None):
        super().__init__(name, assert_held)
        self.expected_id = _integer(expected_id, 'expected_id')
        self.expected_name = _text(expected_name, 'expected_name') if expected_name is not None else None
        self.command = _path(command)
        if self.command not in (Path('/usr/sbin/nvpmodel'), Path('/usr/bin/nvpmodel')):
            raise ValueError('use an explicit system nvpmodel path')
        if not callable(runner):
            raise ValueError('runner must be callable')
        self.runner, self.timeout_s = runner, _positive_time(timeout_s, 'timeout_s')

    def capture(self):
        result = self.runner([str(self.command), '-q'], capture_output=True, text=True,
                             timeout=self.timeout_s, check=False)
        match = re.fullmatch(r'NV Power Mode:\s*([^\n]+)\n\s*(\d+)\s*', result.stdout.strip())
        if result.returncode != 0 or match is None:
            raise ReadinessError('unexpected nvpmodel query response')
        return {'id': int(match.group(2)), 'name': match.group(1).strip()}

    def apply(self, snapshot):
        current = self.capture()
        if current['id'] != self.expected_id or (self.expected_name is not None
                                                and current['name'] != self.expected_name):
            raise ReadinessError('power mode mismatch; prepare externally before experiment')

    def restore(self, snapshot):
        if self.capture() != snapshot:
            raise RestorationError('power mode changed externally; no automatic switching permitted')


class SwappinessControl(FileValueControl):
    def __init__(self, value, *, path='/proc/sys/vm/swappiness', name='swappiness', assert_held=None):
        super().__init__(name, path, _integer(value, 'swappiness', 0, 200), assert_held=assert_held)


class AffinityControl(_Control):
    """Process-leader affinity, inherited by subsequently created worker threads.

    Existing threads keep their own affinity. No other processes are modified.
    """
    def __init__(self, cpus, *, name='affinity', get_affinity=getattr(os, 'sched_getaffinity', None),
                 set_affinity=getattr(os, 'sched_setaffinity', None), assert_held=None):
        super().__init__(name, assert_held)
        if not isinstance(cpus, (set, frozenset, list, tuple)) or not cpus:
            raise ValueError('cpus must be a nonempty CPU collection')
        self.cpus = frozenset(_integer(cpu, 'CPU id') for cpu in cpus)
        if len(self.cpus) != len(cpus):
            raise ValueError('CPU ids must be distinct')
        if get_affinity is not None and not callable(get_affinity) or set_affinity is not None and not callable(set_affinity):
            raise ValueError('affinity callbacks must be callable')
        self.get_affinity, self.set_affinity = get_affinity, set_affinity
        self.pid = os.getpid()

    def capture(self):
        if self.get_affinity is None or self.set_affinity is None:
            raise ReadinessError('process affinity is unavailable on this platform')
        if os.getpid() != self.pid:
            raise ReadinessError('affinity control cannot be reused after fork')
        return sorted(self.get_affinity(self.pid))

    def _set(self, cpus):
        if os.getpid() != self.pid:
            raise ReadinessError('affinity control cannot be reused after fork')
        self._assert()
        self.set_affinity(self.pid, set(cpus))
        if set(self.capture()) != set(cpus):
            raise ReadinessError('process affinity readback mismatch')

    def apply(self, snapshot):
        self._set(self.cpus)

    def restore(self, snapshot):
        self._restore_attempts([('affinity', lambda: self._set(snapshot))])
