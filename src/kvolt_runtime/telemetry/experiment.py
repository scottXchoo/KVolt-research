"""Optional energy measurement around arbitrary experiment code.

Only Python's standard library and the existing read-only INA3221 collector are
used. CUDA completion is supplied by the caller, never inferred by this module.
"""
import json
import math
from pathlib import Path
import re
import time

from ..errors import ContractError
from .metrics import RAILS, SCOPE, energy_metrics
from .power import INAPowerCollector, PowerProfile, RailPaths, samples_csv


def discover_power_profile(hwmon_root='/sys/class/hwmon', *,
                           interval_ns=20_000_000, max_gap_ns=200_000_000):
    """Find the AGX Orin's three independent rails by name, never hwmon index.

    Discovery only reads labels and checks that voltage/current files exist.
    Numeric reads and permission failures are checked by the bounded collector
    startup. Ambiguous/missing rails fail instead of selecting an arbitrary chip.
    These polling/gap defaults are development settings, not sensor qualification.
    """
    root = Path(hwmon_root).resolve()
    rails = {}
    try:
        if not root.is_dir():
            raise ContractError('hwmon directory not found: '+str(root))
        for device in sorted(root.glob('hwmon*')):
            name = device / 'name'
            if not name.is_file() or name.read_text(encoding='utf-8').strip() != 'ina3221':
                continue
            for label_path in sorted(device.glob('in*_label')):
                match = re.fullmatch(r'in([1-3])_label', label_path.name)
                if match is None:
                    continue
                label = label_path.read_text(encoding='utf-8').strip()
                if label not in RAILS:
                    continue
                if label in rails:
                    raise ContractError('ambiguous INA3221 rail: '+label)
                channel = match.group(1)
                current = device / ('curr'+channel+'_input')
                voltage = device / ('in'+channel+'_input')
                if not current.is_file() or not voltage.is_file():
                    raise ContractError('missing INA3221 current/voltage input: '+label)
                rails[label] = RailPaths(str(label_path), str(current), str(voltage))
    except (OSError, UnicodeError) as error:
        raise ContractError('INA3221 discovery failed: '+str(error)) from error
    missing = set(RAILS) - set(rails)
    if missing:
        raise ContractError('missing AGX Orin INA3221 rails: '+', '.join(sorted(missing)))
    return PowerProfile(
        rails=rails, interval_ns=interval_ns, max_gap_ns=max_gap_ns,
        ready_timeout_ns=5_000_000_000, coverage_timeout_ns=2_000_000_000,
        join_timeout_ns=2_000_000_000, name='discovered-agx-orin-development',
        evidence='Read-only INA3221 label discovery; sensor response unvalidated')


def interval_energy_metrics(samples, start_ns, end_ns, *, max_gap_ns):
    """Return joules and average watts for one complete, bracketed interval.

    Reuse the request integrator's strict trace validation and full-interval
    integral. Its single-segment mode avoids introducing token assumptions or
    Prefill/Decode field names into the public experiment result.
    """
    measured = energy_metrics(samples, start_ns, end_ns, end_ns, 1,
                              completed=True, max_gap_ns=max_gap_ns)
    result = dict(valid=False, reason=measured['reason'], duration_s=None,
                  energy_j=None, average_power_w=None, rails={}, scope=SCOPE,
                  start_ns=start_ns, end_ns=end_ns, sensor_response_validated=False,
                  boundary_method=measured['boundary_method'])
    if not measured['valid']:
        return result
    try:
        duration = (end_ns-start_ns)/1e9
        average = measured['total_energy_j']/duration
        rails = {rail: dict(energy_j=values['total_energy_j'],
                            average_power_w=values['total_energy_j']/duration)
                 for rail, values in measured['rails'].items()}
        if not all(math.isfinite(value) for value in
                   [duration, average, *[v['average_power_w'] for v in rails.values()]]):
            raise ValueError('nonfinite_interval_metrics')
    except (OverflowError, ZeroDivisionError, ValueError):
        result['reason'] = 'nonfinite_interval_metrics'
        return result
    result.update(valid=True, reason=None, duration_s=duration,
                  energy_j=measured['total_energy_j'], average_power_w=average, rails=rails)
    return result


class EnergyMeter:
    """One optional measurement, used as a context manager or start()/stop().

    Start waits for fresh samples, then synchronizes before the start boundary.
    Stop synchronizes before the end boundary, then collects a right bracket.
    Sensor failures yield null energy and ``valid=False``. Startup failures and
    synchronization exceptions propagate after cleanup. A workload exception is
    always preserved; its trace is diagnostic and its energy remains null.

    ``enabled=False`` bypasses discovery, collection, clocks and synchronization.
    Use a fresh meter per interval. Spawn requires an import-safe main guard.
    ``clock`` and ``collector_factory`` allow controlled tests/custom collectors
    implementing the INAPowerCollector interface in the same monotonic domain.
    """
    def __init__(self, profile=None, *, enabled=True, synchronize=None,
                 clock=time.monotonic_ns, collector_factory=INAPowerCollector):
        if profile is not None and not isinstance(profile, PowerProfile):
            raise ContractError('EnergyMeter profile must be PowerProfile or None')
        if type(enabled) is not bool:
            raise ContractError('EnergyMeter enabled must be bool')
        if synchronize is not None and not callable(synchronize):
            raise ContractError('synchronize must be callable or None')
        if not callable(clock) or not callable(collector_factory):
            raise ContractError('clock and collector_factory must be callable')
        self.profile = profile
        self.enabled = enabled
        self.synchronize = synchronize
        self.clock = clock
        self._collector_factory = collector_factory
        self._collector = None
        self._started = False
        self._start_ns = None
        self._end_ns = None
        self._errors = []
        self.result = None
        self.receipt = None

    def _error(self, stage, error):
        self._errors.append(dict(stage=stage, type=type(error).__name__, message=str(error)))

    def _timestamp(self):
        value = self.clock()
        if type(value) is not int or value < 0:
            raise ContractError('clock must return nonnegative integer monotonic ns')
        return value

    def _cleanup(self):
        if self._collector is not None:
            try:
                self.receipt = self._collector.stop()
            except BaseException as error:
                self._error('collector_stop', error)
                return error
        return None

    def _result(self, reason, completed):
        result = dict(valid=False, reason=reason, duration_s=None, energy_j=None,
                      average_power_w=None, rails={}, start_ns=self._start_ns,
                      end_ns=self._end_ns, completed=completed, scope=SCOPE,
                      sensor_response_validated=False, clock='time.monotonic_ns'
                      if self.clock is time.monotonic_ns else 'injected-clock',
                      errors=list(self._errors), collector=None)
        if self.receipt is not None:
            result['collector'] = {key: value for key, value in self.receipt.items()
                                   if key not in ('samples', 'raw_trace_jsonl')}
            result['errors'].extend(self.receipt.get('errors', []))
        if self._start_ns is not None and self._end_ns is not None:
            result['duration_s'] = (self._end_ns-self._start_ns)/1e9
        self.result = result
        return result

    def start(self):
        """Start once; fail before running the workload if sensors are not ready."""
        if self._started:
            raise ContractError('EnergyMeter cannot start twice; use a new meter')
        self._started = True
        if not self.enabled:
            return self
        try:
            if self.profile is None:
                self.profile = discover_power_profile()
            self._collector = self._collector_factory(self.profile, clock=self.clock)
            ready = self._collector.start()
            if not ready.get('ready'):
                raise ContractError('energy collector not ready: '+str(ready.get('errors', [])))
            if self.synchronize is not None:
                self.synchronize()
            self._start_ns = self._timestamp()
        except BaseException as error:
            self._error('start', error)
            self._cleanup()
            self._result('start_failed', False)
            raise
        return self

    def stop(self, *, completed=True):
        """Finish once; repeated calls return the same result without new reads."""
        if type(completed) is not bool:
            raise ContractError('completed must be bool')
        if not self._started:
            raise ContractError('EnergyMeter has not started')
        if self.result is not None:
            return self.result
        if self._collector is None:
            return self._result('not_collected', completed)
        boundary_error = interrupt = None
        try:
            if self.synchronize is not None:
                self.synchronize()
            self._end_ns = self._timestamp()
        except BaseException as error:
            boundary_error = error
            self._error('stop_boundary', error)
        try:
            if self._end_ns is not None:
                self.receipt = self._collector.finish(self._end_ns)
        except BaseException as error:
            self._error('collector_finish', error)
            if not isinstance(error, Exception):
                interrupt = error
        finally:
            # INAPowerCollector.stop is idempotent, including after finish.
            cleanup_error = self._cleanup()
            if cleanup_error is not None and not isinstance(cleanup_error, Exception):
                interrupt = interrupt or cleanup_error
        reason = ('workload_failed' if not completed else
                  'synchronization_failed' if boundary_error else
                  'collector_error' if self._errors or self.receipt is None or
                  self.receipt.get('status') != 'stopped' or
                  self.receipt.get('errors') else None)
        result = self._result(reason, completed)
        if reason is None:
            measured = interval_energy_metrics(
                self.receipt['samples'], self._start_ns, self._end_ns,
                max_gap_ns=self.profile.max_gap_ns)
            result.update(measured)
        if boundary_error is not None and completed:
            raise boundary_error
        if interrupt is not None and completed:
            raise interrupt
        return result

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.stop(completed=exc_type is None)
        return False

    def save(self, output_dir):
        """Save summary, raw CSV, and exact worker JSONL outside the interval.

        Existing artifact names are never overwritten. Failures propagate;
        partial output remains inspectable. Disabled runs save empty raw traces.
        """
        if self.result is None:
            raise ContractError('finish EnergyMeter before saving')
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths = {name: root/name for name in
                 ('energy.json', 'power-samples.csv', 'power-trace.jsonl')}
        if any(path.exists() for path in paths.values()):
            raise FileExistsError('energy artifacts already exist in '+str(root))
        receipt = self.receipt or {}
        payloads = {
            'power-trace.jsonl': receipt.get('raw_trace_jsonl', ''),
            'power-samples.csv': samples_csv(receipt.get('samples', [])),
            'energy.json': json.dumps(self.result, ensure_ascii=False,
                                      allow_nan=False, indent=2)+'\n',
        }
        for name, payload in payloads.items():
            with paths[name].open('x', encoding='utf-8', newline='') as stream:
                stream.write(payload)
        return paths
