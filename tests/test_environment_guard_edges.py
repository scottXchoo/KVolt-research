"""Regression scenarios found by the independent environment lifecycle review."""
import os
import threading
import time

import pytest

from experiment_control.contracts import EnvironmentError, MetricRule, Reading, ReadinessError
from experiment_control.guard import Environment
from experiment_control.lock import FileLock


class SequenceProbe:
    name = 'sensor'

    def __init__(self, values):
        self.values = iter(values)
        self.last = None

    def sample(self):
        self.last = next(self.values, self.last)
        return Reading({'value': self.last})


def test_baseline_must_satisfy_conditions_before_workload_starts():
    calls = []
    env = Environment(probes=[SequenceProbe([40, 80, 80])],
                      rules=[MetricRule('sensor', 'value', maximum=60)], sample_interval_s=10)
    with pytest.raises(ReadinessError):
        env.run(lambda: calls.append('workload'))
    assert calls == []
    assert env.report['workload']['status'] == 'not_started'


def test_close_failure_preserves_original_workload_exception(tmp_path):
    class FailingClose:
        def __init__(self, stream):
            self.stream = stream
        def write(self, value):
            return self.stream.write(value)
        def flush(self):
            return self.stream.flush()
        def close(self):
            self.stream.close()
            raise OSError('simulated close failure')

    env = Environment(output_dir=tmp_path/'run')
    with pytest.raises(KeyError, match='original workload failure'):
        with env:
            env._stream = FailingClose(env._stream)
            raise KeyError('original workload failure')
    assert env.report['workload']['error']['type'] == 'KeyError'
    assert env.report.get('recording_error')


def test_one_slow_sample_is_not_a_stable_temperature_interval():
    now = [0]
    class SlowProbe:
        name = 'sensor'
        def sample(self):
            now[0] += 2_000_000_000
            return Reading({'value': 40})
    calls = []
    env = Environment(probes=[SlowProbe()], rules=[MetricRule('sensor', 'value', max_span=1)],
                      clock_ns=lambda: now[0], stable_for_s=1, ready_timeout_s=1,
                      sample_interval_s=.1, sleep=lambda seconds: None)
    with pytest.raises(ReadinessError):
        env.run(lambda: calls.append('workload'))
    assert calls == []


def test_counter_decrease_above_initial_baseline_is_a_violation():
    env = Environment(probes=[SequenceProbe([100, 100, 120, 110])],
                      rules=[MetricRule('sensor', 'value', max_increase=100)],
                      sample_interval_s=.001)
    with env:
        deadline = time.monotonic() + 2
        while env.report['monitor']['samples'] < 4 and time.monotonic() < deadline:
            time.sleep(.001)
        assert env.report['monitor']['samples'] >= 4
    assert any(v['reason'] == 'counter_decreased'
               for v in env.report['environment']['violations'])


def test_late_probe_completion_cannot_mutate_final_report(tmp_path):
    started, release, completed = threading.Event(), threading.Event(), threading.Event()
    class BlockingProbe:
        name = 'sensor'
        def __init__(self):
            self.calls = 0
        def sample(self):
            self.calls += 1
            if self.calls == 3:
                started.set()
                release.wait(10)
                completed.set()
                return Reading({'value': 999})
            return Reading({'value': 40})

    env = Environment(probes=[BlockingProbe()], rules=[MetricRule('sensor', 'value', maximum=60)],
                      output_dir=tmp_path/'run', sample_interval_s=.001)
    try:
        with pytest.raises(EnvironmentError):
            with env:
                assert started.wait(2), 'monitor did not begin the blocking read'
        import copy
        frozen = copy.deepcopy(env.report)
        release.set()
        assert completed.wait(2), 'blocking probe did not finish'
        env._thread.join(timeout=2)
        assert not env._thread.is_alive()
        assert env.report == frozen
    finally:
        release.set()
        if env._thread is not None:
            env._thread.join(timeout=2)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires POSIX fork')
def test_fork_child_close_does_not_unlock_parent_file_lock(tmp_path):
    path = tmp_path/'hardware.lock'
    path.touch()
    owner = FileLock(path)
    with owner:
        child = os.fork()
        if child == 0:
            try:
                owner.__exit__(None, None, None)
            finally:
                os._exit(0)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        contender = FileLock(path)
        try:
            with pytest.raises(BlockingIOError):
                contender.__enter__()
        finally:
            contender.__exit__(None, None, None)
        assert owner.held


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires POSIX fork')
@pytest.mark.parametrize('primary_error', [False, True])
def test_fork_child_environment_cannot_restore_parent_controls(tmp_path, monkeypatch, primary_error):
    # Avoid forking with an actual Python monitor thread running. Lifecycle and
    # inherited lock behavior remain real; this test does not test sampling.
    class DormantThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return False
    monkeypatch.setattr('experiment_control.guard.threading.Thread', DormantThread)
    marker = tmp_path/'restored'
    class Control:
        name = 'test'
        def capture(self):
            return 1
        def apply(self, snapshot):
            pass
        def restore(self, snapshot):
            marker.write_text(str(os.getpid()))
    path = tmp_path/'hardware.lock'
    path.touch()
    env = Environment(controls=[Control()], lock=FileLock(path), output_dir=tmp_path/'run')
    with env:
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            try:
                try:
                    error = KeyError('child workload') if primary_error else None
                    outcome = env.__exit__(type(error) if error is not None else None, error, None)
                except EnvironmentError:
                    os.write(write_fd, b'blocked')
                else:
                    os.write(write_fd, b'preserved' if primary_error and outcome is False else b'allowed')
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        try:
            result = os.read(read_fd, 20)
        finally:
            os.close(read_fd)
        os.waitpid(child, 0)
        assert result == (b'preserved' if primary_error else b'blocked')
        assert not marker.exists(), 'fork child restored parent hardware state'
        contender = FileLock(path)
        try:
            with pytest.raises(BlockingIOError):
                contender.__enter__()
        finally:
            contender.__exit__(None, None, None)
    assert marker.read_text() == str(os.getpid())
