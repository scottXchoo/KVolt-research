"""Transactional environment scope; no inference-framework imports or policies."""
from dataclasses import asdict
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time

from .contracts import (EnvironmentError, MetricRule, Reading, ReadinessError,
                        RestorationError, ViolationError, finite_number)


def _error(error):
    result = {'type': type(error).__name__, 'message': str(error)}
    if type(getattr(error, 'signum', None)) is int:
        result['signum'] = error.signum
    return result


class Environment:
    """Single-use scope. Configuration changes happen only before the workload.

    Custom probes must finish promptly and only read state. ``synchronize`` must
    wait for asynchronous work before restoration. Runtime violations are recorded
    by default; ``raise`` also raises at checkpoint/exit, never retries a workload.
    """

    def __init__(self, *, controls=(), probes=(), rules=(), output_dir=None,
                 lock=None, borrow_lock=False, warmup=None, warmup_runs=0,
                 synchronize=None, sample_interval_s=0.2, ready_timeout_s=0,
                 stable_for_s=0, on_violation='record',
                 metadata=None, clock_ns=time.monotonic_ns, sleep=time.sleep):
        self.controls, self.probes, self.rules = tuple(controls), tuple(probes), tuple(rules)
        for collection in (self.controls, self.probes):
            names = [x.name for x in collection]
            if any(type(n) is not str or not n for n in names) or len(names) != len(set(names)):
                raise ValueError('control/probe names must be nonempty and unique')
        if any(not isinstance(r, MetricRule) for r in self.rules):
            raise ValueError('rules must be MetricRule instances')
        if any(r.probe not in {p.name for p in self.probes} for r in self.rules):
            raise ValueError('rule references an unknown probe')
        finite_number(sample_interval_s, 'sample_interval_s', minimum=0.001)
        finite_number(ready_timeout_s, 'ready_timeout_s', minimum=0)
        finite_number(stable_for_s, 'stable_for_s', minimum=0)
        if stable_for_s > ready_timeout_s:
            raise ValueError('stable_for_s exceeds ready_timeout_s')
        if type(warmup_runs) is not int or warmup_runs < 0:
            raise ValueError('warmup_runs must be a nonnegative integer')
        if warmup is not None and not callable(warmup) or synchronize is not None and not callable(synchronize):
            raise ValueError('warmup/synchronize must be callable')
        if warmup_runs and warmup is None:
            raise ValueError('warmup_runs requires a warmup callback')
        if on_violation not in ('record', 'raise'):
            raise ValueError('on_violation must be record or raise')
        if type(borrow_lock) is not bool or borrow_lock and lock is None:
            raise ValueError('borrow_lock requires a lock')
        if self.controls and (lock is None or output_dir is None):
            raise ValueError('controls require a hardware lock and durable output_dir')
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.lock, self.borrow_lock = lock, borrow_lock
        self.warmup, self.warmup_runs, self.synchronize = warmup, warmup_runs, synchronize
        self.sample_interval_s, self.ready_timeout_s, self.stable_for_s = sample_interval_s, ready_timeout_s, stable_for_s
        self.on_violation, self.clock_ns, self.sleep = on_violation, clock_ns, sleep
        self._used = self._entered = self._owns_lock = False
        self._pid = None
        self._accept_samples = True
        self._applied, self._snapshots = [], []
        self._stream = self._thread = None
        self._stop = threading.Event()
        self._sample_lock, self._journal_lock = threading.Lock(), threading.Lock()
        self._state_lock = threading.RLock()
        self._baseline = None
        self._previous_run = None
        self._ready_window = ()
        self.report = {
            'schema': 'experiment-control.v1', 'clock': 'time.monotonic_ns',
            'start_ns': None, 'end_ns': None, 'rules': [asdict(r) for r in self.rules],
            'metadata': json.loads(json.dumps(metadata or {}, allow_nan=False)),
            'preparation': {'status': 'not_started', 'warmup_completed': 0},
            'workload': {'status': 'not_started'},
            'environment': {'status': 'not_evaluated', 'violations': [], 'violation_count': 0},
            'monitor': {'samples': 0, 'read_errors': 0, 'max_gap_ns': 0, 'max_read_ns': 0,
                        'interval_s': sample_interval_s, 'errors': []},
            'restoration': {'status': 'not_needed', 'errors': []},
            'limitations': ['sampled observations cannot prove absence of transient hardware throttling',
                            'environment status does not grant SLO, energy, or formal qualification'],
        }

    def _journal(self, kind, **data):
        row = {'kind': kind, 'timestamp_ns': self.clock_ns(), **data}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
        if self._stream is not None:
            with self._journal_lock:
                if self._stream is not None:
                    self._stream.write(encoded + '\n')
                    self._stream.flush()

    def _write_json(self, name, data):
        if self.output_dir is not None:
            encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
            temp = self.output_dir / (name + '.tmp')
            with temp.open('w', encoding='utf-8') as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(self.output_dir / name)

    def _sample(self, phase):
        with self._sample_lock:
            start = self.clock_ns()
            readings = {}
            for probe in self.probes:
                try:
                    reading = probe.sample()
                    if not isinstance(reading, Reading):
                        raise TypeError('probe.sample must return Reading')
                except Exception as error:
                    reading = Reading({}, {'probe': f'{type(error).__name__}: {error}'})
                readings[probe.name] = {'values': dict(reading.values), 'errors': dict(reading.errors)}
            end = self.clock_ns()
            if end < start:
                raise EnvironmentError('monotonic clock moved backwards')
            sample = {'phase': phase, 'read_start_ns': start, 'read_end_ns': end, 'probes': readings}
            with self._state_lock:
                if not self._accept_samples:
                    return sample
                monitor = self.report['monitor']
                previous = monitor.get('last_sample_ns')
                if previous is not None:
                    monitor['max_gap_ns'] = max(monitor['max_gap_ns'], start - previous)
                monitor['last_sample_ns'] = start
                monitor['max_read_ns'] = max(monitor['max_read_ns'], end - start)
                monitor['samples'] += 1
                monitor['read_errors'] += sum(len(r['errors']) for r in readings.values())
                monitor['last_sample'] = sample
                self._journal('sample', **sample)
            return sample

    @staticmethod
    def _value(sample, rule):
        section = sample['probes'].get(rule.probe, {})
        values, errors = section.get('values', {}), section.get('errors', {})
        value = values.get(rule.metric)
        if rule.metric in errors:
            return None
        return value

    def _evaluate(self, sample, phase, window=()):
        failures = []
        for rule in self.rules:
            if phase not in rule.phases:
                continue
            value = self._value(sample, rule)
            reason = None
            if value is None:
                reason = 'unavailable' if rule.required else None
            elif rule.expected is not None and (type(value) is not type(rule.expected) or value != rule.expected):
                reason = 'unexpected_value'
            elif any(v is not None for v in (rule.minimum, rule.maximum, rule.max_increase, rule.max_span)):
                if type(value) not in (int, float):
                    reason = 'not_numeric'
                elif rule.minimum is not None and value < rule.minimum:
                    reason = 'below_minimum'
                elif rule.maximum is not None and value > rule.maximum:
                    reason = 'above_maximum'
                elif phase == 'run' and rule.max_increase is not None:
                    baseline = self._value(self._baseline, rule) if self._baseline is not None else None
                    previous = self._value(self._previous_run, rule) if self._previous_run is not None else baseline
                    if type(baseline) not in (int, float):
                        reason = 'baseline_unavailable'
                    elif value < baseline or type(previous) in (int, float) and value < previous:
                        reason = 'counter_decreased'
                    elif value - baseline > rule.max_increase:
                        reason = 'increase_exceeded'
                if reason is None and phase == 'ready' and rule.max_span is not None:
                    history = [self._value(s, rule) for s in window] + [value]
                    if all(type(v) in (int, float) for v in history) and max(history) - min(history) > rule.max_span:
                        reason = 'unstable'
            if reason:
                failures.append({'probe': rule.probe, 'metric': rule.metric, 'value': value,
                                 'phase': phase, 'reason': reason, 'timestamp_ns': sample['read_end_ns']})
        return failures

    def _add_violations(self, failures):
        with self._state_lock:
            if not self._accept_samples:
                return
            state = self.report['environment']
            state['violation_count'] += len(failures)
            state['violations'].extend(failures[:max(0, 1000 - len(state['violations']))])
            if failures:
                state['status'] = 'violated'
                self._journal('violations', violations=failures)

    def _run_sample(self):
        sample = self._sample('run')
        with self._state_lock:
            if self._accept_samples:
                self._add_violations(self._evaluate(sample, 'run'))
                self._previous_run = sample

    def _ready(self):
        start, window = self.clock_ns(), []
        while True:
            sample = self._sample('ready')
            failures = self._evaluate(sample, 'ready', window)
            window = [] if failures else window + [sample]
            elapsed = (sample['read_end_ns'] - start) / 1e9
            stable = bool(window) and (self.stable_for_s == 0 or len(window) >= 2 and
                     (sample['read_end_ns'] - window[0]['read_end_ns']) / 1e9 >= self.stable_for_s)
            deadline_passed = self.ready_timeout_s > 0 and elapsed > self.ready_timeout_s
            if not failures and stable and not deadline_passed:
                self._ready_window = tuple(window)
                self._journal('ready', stable_for_s=self.stable_for_s)
                return
            if elapsed >= self.ready_timeout_s:
                if not failures:
                    failures = [{'phase': 'ready', 'reason': 'stability_timeout', 'timestamp_ns': sample['read_end_ns']}]
                self._add_violations(failures)
                raise ReadinessError('environment readiness conditions were not met')
            self.sleep(min(self.sample_interval_s, self.ready_timeout_s - elapsed))

    def _monitor(self):
        while not self._stop.wait(self.sample_interval_s):
            try:
                self._run_sample()
            except Exception as error:
                with self._state_lock:
                    if self._accept_samples:
                        self.report['monitor']['errors'].append(_error(error))
                self._stop.set()

    def checkpoint(self):
        """Cooperative check; does not sample, modify clocks, or retry work."""
        if not self._entered:
            raise EnvironmentError('environment is not active')
        if self._pid != os.getpid():
            raise EnvironmentError('environment cannot be used by a forked child')
        if self.lock is not None:
            self.lock.assert_held()
        if self.report['monitor']['errors']:
            raise EnvironmentError('environment monitor failed')
        if self.on_violation == 'raise' and self.report['environment']['violation_count']:
            raise ViolationError('environment conditions were violated; inspect report')

    def __enter__(self):
        if self._used:
            raise EnvironmentError('environment scopes are single-use')
        self._used = True
        self._pid = os.getpid()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self._stream = (self.output_dir/'events.jsonl').open('x', encoding='utf-8')
        self.report['start_ns'] = self.clock_ns()
        try:
            if self.lock is not None:
                if not self.borrow_lock:
                    self.lock.__enter__()
                    self._owns_lock = True
                self.lock.assert_held()
            for control in self.controls:
                state = control.capture()
                # Validate and detach snapshots before any mutation can happen.
                state = json.loads(json.dumps(state, allow_nan=False))
                self._snapshots.append((control, state))
            self._write_json('snapshot.json', {'schema': 'experiment-control.snapshot.v1',
                'controls': [{'name': c.name, 'state': s} for c, s in self._snapshots]})
            for control, state in self._snapshots:
                self.lock.assert_held()
                self._applied.append((control, state))
                self._journal('apply_started', control=control.name, snapshot=state)
                control.apply(state)
                self._journal('apply_finished', control=control.name)
            for _ in range(self.warmup_runs):
                self.warmup()
                if self.synchronize is not None:
                    self.synchronize()
                self.report['preparation']['warmup_completed'] += 1
                self._journal('warmup_finished', iteration=self.report['preparation']['warmup_completed'])
            self._ready()
            self._baseline = self._sample('baseline')
            failures = self._evaluate(self._baseline, 'ready', self._ready_window) + self._evaluate(self._baseline, 'run')
            if failures:
                self._add_violations(failures)
                raise ReadinessError('environment changed or required counters are unavailable before workload')
            self._previous_run = self._baseline
            self.report['environment']['status'] = 'satisfied' if self.rules else 'not_evaluated'
            self.report['preparation']['status'] = 'succeeded'
            self.report['workload'] = {'status': 'running', 'start_ns': self.clock_ns()}
            self._entered = True
            self._thread = threading.Thread(target=self._monitor, name='experiment-environment', daemon=True)
            self._thread.start()
            return self
        except BaseException as error:
            self.report['preparation'].update(status='failed', error=_error(error))
            if self.warmup_runs and self.synchronize is not None:
                try:
                    self.synchronize()
                except BaseException as sync_error:
                    self.report['preparation']['synchronize_error'] = _error(sync_error)
            self._cleanup()
            raise

    def _cleanup(self):
        with self._state_lock:
            self._accept_samples = False
        errors = []
        for control, snapshot in reversed(self._applied):
            try:
                self.lock.assert_held()
                control.restore(snapshot)
                self._journal('restored', control=control.name)
            except BaseException as error:
                errors.append({'control': control.name, **_error(error)})
        self._applied.clear()
        self.report['restoration'] = {'status': 'failed' if errors else 'succeeded', 'errors': errors}
        if self._owns_lock:
            try:
                self.lock.__exit__(None, None, None)
            except BaseException as error:
                self.report['restoration']['errors'].append({'control': 'lock', **_error(error)})
                self.report['restoration']['status'] = 'failed'
            self._owns_lock = False
        self.report['end_ns'] = self.clock_ns()
        with self._journal_lock:
            if self._stream is not None:
                try:
                    self._stream.close()
                except BaseException as error:
                    self.report['recording_error'] = _error(error)
                self._stream = None
        try:
            self._write_json('summary.json', self.report)
        except BaseException as error:
            self.report['recording_error'] = _error(error)

    def __exit__(self, exc_type, exc, tb):
        if self._pid != os.getpid():
            # A child must not restore global controls, close a borrowed lock,
            # or publish into its parent's artifact files.
            if exc is not None:
                return False
            raise EnvironmentError('a forked child cannot close its parent environment')
        sync_error = None
        try:
            if self.synchronize is not None:
                self.synchronize()
        except BaseException as error:
            sync_error = error
            self.report['workload']['synchronize_error'] = _error(error)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_s * 2))
            if self._thread.is_alive():
                self.report['monitor']['errors'].append({'type': 'TimeoutError', 'message': 'probe did not stop promptly'})
        try:
            if self._thread is None or not self._thread.is_alive():
                self._run_sample()
        except BaseException as error:
            self.report['monitor']['errors'].append(_error(error))
        self.report['workload'].update(status='failed' if exc is not None or sync_error is not None else 'succeeded',
                                       end_ns=self.clock_ns())
        if exc is not None:
            self.report['workload']['error'] = _error(exc)
        self._entered = False
        self._cleanup()
        if exc is not None:
            return False
        if sync_error is not None:
            raise sync_error
        if self.report['restoration']['status'] == 'failed':
            raise RestorationError('environment restoration failed; inspect report')
        if self.report.get('recording_error') or self.report['monitor']['errors']:
            raise EnvironmentError('environment recording/monitor failed; inspect report')
        if self.on_violation == 'raise' and self.report['environment']['violation_count']:
            raise ViolationError('environment conditions were violated; inspect report')
        return False

    def run(self, function, /, *args, **kwargs):
        with self:
            return function(*args, **kwargs)


def controlled(factory):
    """Decorate a callable using a fresh Environment from a zero-argument factory."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with factory():
                return function(*args, **kwargs)
        return wrapped
    return decorate
