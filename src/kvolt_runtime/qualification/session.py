"""Explicit experiment preparation for M3 development qualification only.

This helper is outside the Runtime DeviceController. It temporarily prepares
CPU/EMC/fan, never changes MAXN, and restores its captured controls in finally.
The caller must own the same common lock throughout preparation and validation.
"""
from pathlib import Path
import subprocess

from ..device import SysfsIO
from ..execution.contracts import ExecutionError, error_record


class ExperimentIO(SysfsIO):
    def fan_service(self, command, action, timeout_ns):
        if action not in ('start', 'stop'):
            raise ValueError('only fan service start/stop is supported')
        subprocess.run([str(command), action, 'nvfancontrol'], check=True,
                       capture_output=True, text=True, timeout=timeout_ns / 1e9)


class ExperimentPreparation:
    """Not part of normal run. Requires the explicit --prepare-experiment flag."""

    def __init__(self, profile, *, lock, io=None):
        self.profile, self.lock = profile, lock
        self.io = io if io is not None else ExperimentIO()
        self.snapshot = None
        self.changes = []
        self.mutation_attempted = False
        self.phase = None

    def _integer(self, path):
        value = self.io.read(path)
        if not value.isdecimal():
            raise ValueError(f'invalid integer preparation control: {path}')
        return int(value)

    def capture(self):
        self.lock.assert_held()
        p = self.profile
        mode = self.io.query_maxn(p.maxn_command, p.query_timeout_ns)
        if mode['name'] != 'MAXN' or mode['mode_id'] != p.maxn_mode_id or self.io.read(p.tpc_path) != p.tpc_expected:
            raise ValueError('MAXN/TPC must already be prepared; this tool does not switch power modes')
        if self._integer(p.emc_cap_path) != 3199000000 or self._integer(p.emc_max_rate_path) != 3199000000:
            raise ValueError('expected EMC maximum is not available')
        cpu = []
        for path in p.cpu_policy_paths:
            cpu.append({'path': str(path), 'minimum': self._integer(path / 'scaling_min_freq'),
                        'maximum': self._integer(path / 'scaling_max_freq'),
                        'governor': self.io.read(path / 'scaling_governor')})
        service = self.io.query_fan_service(p.systemctl_command, p.query_timeout_ns)
        self.snapshot = {'cpu': cpu, 'emc': {'rate': self._integer(p.emc_rate_path),
                         'lock': self._integer(p.emc_lock_path), 'halt': self._integer(p.emc_halt_path)},
                         'fan_pwm': self._integer(p.fan_path),
                         'fan_policy': self.io.read(p.fan_policy_path), 'fan_service': service,
                         'mode': mode, 'tpc': self.io.read(p.tpc_path)}
        return self.snapshot

    def _write(self, path, value):
        self.lock.assert_held()
        self.mutation_attempted = True
        row = {'path': str(path), 'value': str(value), 'phase': self.phase, 'succeeded': False}
        self.changes.append(row)
        self.io.write(path, value)
        row['succeeded'] = True

    def prepare(self):
        if self.snapshot is None:
            raise ValueError('capture preparation state before changing controls')
        self.phase = 'prepare'
        p = self.profile
        # Same order as the existing P2 preparation, without configuration edits.
        for path, value in ((p.emc_lock_path, 1), (p.emc_halt_path, 1), (p.emc_rate_path, 3199000000)):
            self._write(path, value)
        self.mutation_attempted = True
        self.io.fan_service(p.systemctl_command, 'stop', p.query_timeout_ns)
        self._write(p.fan_policy_path, 'user_space')
        self._write(p.fan_path, 255)
        for path in p.cpu_policy_paths:
            current_max = self._integer(path / 'scaling_max_freq')
            if current_max < 2201600:
                self._write(path / 'scaling_max_freq', 2201600)
            self._write(path / 'scaling_min_freq', 2201600)
            self._write(path / 'scaling_max_freq', 2201600)
        return {'changes': [dict(row) for row in self.changes],
                'purpose': 'external_experiment_preparation_not_runtime'}

    def restore(self):
        if self.snapshot is None or not self.mutation_attempted:
            return {'verified': True, 'changed': False, 'errors': []}
        self.phase = 'restore'
        p, old = self.profile, self.snapshot
        errors = []

        def attempt(work, stage):
            try:
                work()
            except Exception as error:
                errors.append(error_record(error, stage=stage))

        def cpu_restore(row):
            path = Path(row['path'])
            current_max = self._integer(path / 'scaling_max_freq')
            controls = [('scaling_max_freq', row['maximum']), ('scaling_min_freq', row['minimum'])] \
                if row['minimum'] > current_max else [('scaling_min_freq', row['minimum']), ('scaling_max_freq', row['maximum'])]
            for name, value in controls:
                attempt(lambda name=name, value=value: self._write(path / name, value),
                        'external_cpu_control_restore')
            if (self._integer(path / 'scaling_min_freq') != row['minimum'] or
                    self._integer(path / 'scaling_max_freq') != row['maximum'] or
                    self.io.read(path / 'scaling_governor') != row['governor']):
                raise ValueError('CPU preparation restoration mismatch')

        for row in old['cpu']:
            attempt(lambda row=row: cpu_restore(row), 'external_cpu_restore')

        def emc_restore():
            for path, value in ((p.emc_lock_path, 1), (p.emc_halt_path, 1),
                                (p.emc_rate_path, old['emc']['rate']),
                                (p.emc_halt_path, old['emc']['halt']), (p.emc_lock_path, old['emc']['lock'])):
                attempt(lambda path=path, value=value: self._write(path, value),
                        'external_emc_control_restore')
            if self._integer(p.emc_lock_path) != old['emc']['lock'] or self._integer(p.emc_halt_path) != old['emc']['halt']:
                raise ValueError('EMC preparation restoration mismatch')
            if old['emc']['lock'] == 1 and self._integer(p.emc_rate_path) != old['emc']['rate']:
                raise ValueError('locked EMC rate restoration mismatch')
        attempt(emc_restore, 'external_emc_restore')

        def fan_restore():
            attempt(lambda: self._write(p.fan_policy_path, old['fan_policy']), 'external_fan_policy_restore')
            attempt(lambda: self._write(p.fan_path, old['fan_pwm']), 'external_fan_pwm_restore')
            attempt(lambda: self.io.fan_service(p.systemctl_command,
                    'start' if old['fan_service'] == 'active' else 'stop', p.query_timeout_ns),
                    'external_fan_service_restore')
            service = self.io.query_fan_service(p.systemctl_command, p.query_timeout_ns)
            if service != old['fan_service'] or self.io.read(p.fan_policy_path) != old['fan_policy']:
                raise ValueError('fan service/policy restoration mismatch')
            # Active fan control may immediately choose a different PWM.
            if service == 'inactive' and self._integer(p.fan_path) != old['fan_pwm']:
                raise ValueError('inactive fan PWM restoration mismatch')
        attempt(fan_restore, 'external_fan_restore')
        return {'verified': not errors, 'changed': True, 'errors': errors,
                'dynamic_emc_rate_restored_exactly': old['emc']['lock'] == 1,
                'dynamic_fan_pwm_restored_exactly': old['fan_service'] == 'inactive'}
