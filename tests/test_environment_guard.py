import json
import time

import pytest

from experiment_control.contracts import Reading, MetricRule, ReadinessError, RestorationError, ViolationError
from experiment_control.guard import Environment, controlled


class Lock:
    def __init__(self):
        self.held = False
        self.exits = 0
    def __enter__(self):
        self.held = True
        return self
    def assert_held(self):
        assert self.held
    def __exit__(self, *_):
        self.held = False
        self.exits += 1


class Control:
    def __init__(self, name, events, fail_apply=False, fail_restore=False):
        self.name, self.events = name, events
        self.fail_apply, self.fail_restore = fail_apply, fail_restore
    def capture(self):
        self.events.append('capture:' + self.name)
        return {'value': 3}
    def apply(self, state):
        self.events.append('apply:' + self.name)
        if self.fail_apply:
            raise ValueError('partial apply')
    def restore(self, state):
        self.events.append('restore:' + self.name)
        if self.fail_restore:
            raise OSError('restore failed')


class Probe:
    name = 'test'
    def __init__(self, values):
        self.values, self.index = values, 0
    def sample(self):
        value = self.values[min(self.index, len(self.values)-1)]
        self.index += 1
        return Reading({'temperature': value})


def test_transaction_order_and_detach(tmp_path):
    events, lock = [], Lock()
    env = Environment(controls=[Control('a', events), Control('b', events)],
                      output_dir=tmp_path/'run', lock=lock, warmup=lambda: events.append('warmup'),
                      warmup_runs=2, synchronize=lambda: events.append('sync'))
    with env:
        events.append('work')
        assert (tmp_path/'run'/'snapshot.json').is_file()
    assert events[:4] == ['capture:a','capture:b','apply:a','apply:b']
    assert events[-3:] == ['sync','restore:b','restore:a']
    assert env.report['workload']['status'] == 'succeeded'
    assert env.report['restoration']['status'] == 'succeeded'
    assert lock.exits == 1
    assert json.loads((tmp_path/'run'/'summary.json').read_text())['schema'] == 'experiment-control.v1'


def test_partial_apply_rolls_back_all_attempted(tmp_path):
    events = []
    env = Environment(controls=[Control('a',events),Control('b',events,fail_apply=True),Control('c',events)],
                      output_dir=tmp_path/'run',lock=Lock())
    with pytest.raises(ValueError, match='partial apply'):
        with env:
            pytest.fail('must not start')
    assert events[-2:] == ['restore:b','restore:a']
    assert 'apply:c' not in events
    assert env.report['workload']['status'] == 'not_started'


def test_restore_error_preserves_workload_exception(tmp_path):
    env = Environment(controls=[Control('a',[],fail_restore=True)],output_dir=tmp_path/'run',lock=Lock())
    with pytest.raises(KeyError):
        with env:
            raise KeyError('workload')
    assert env.report['restoration']['status'] == 'failed'
    assert env.report['workload']['error']['type'] == 'KeyError'


def test_restore_failure_after_success_is_raised(tmp_path):
    env = Environment(controls=[Control('a',[],fail_restore=True)],output_dir=tmp_path/'run',lock=Lock())
    with pytest.raises(RestorationError):
        env.run(lambda: 7)
    assert env.report['workload']['status'] == 'succeeded'


def test_borrowed_lock_is_not_released(tmp_path):
    lock = Lock()
    with lock:
        with Environment(controls=[Control('a',[])],output_dir=tmp_path/'run',lock=lock,borrow_lock=True):
            pass
        assert lock.held and lock.exits == 0


def test_readiness_missing_and_timeout(tmp_path):
    env = Environment(probes=[Probe([None])],rules=[MetricRule('test','temperature',maximum=60)],
                      output_dir=tmp_path/'run')
    with pytest.raises(ReadinessError):
        env.run(lambda: pytest.fail('not ready'))
    assert env.report['workload']['status'] == 'not_started'


def test_warmup_failure_still_restores(tmp_path):
    events = []
    def fail():
        raise RuntimeError('warmup')
    env = Environment(controls=[Control('a',events)],output_dir=tmp_path/'run',lock=Lock(),warmup=fail,warmup_runs=1)
    with pytest.raises(RuntimeError,match='warmup'):
        env.run(lambda: None)
    assert events[-1] == 'restore:a'


def test_record_and_raise_violations_are_explicit(tmp_path):
    for mode in ('record','raise'):
        env = Environment(probes=[Probe([40,40,80])],rules=[MetricRule('test','temperature',maximum=60)],
                          output_dir=tmp_path/mode,on_violation=mode,sample_interval_s=10)
        if mode == 'raise':
            with pytest.raises(ViolationError):
                env.run(lambda: None)
        else:
            env.run(lambda: None)
        assert env.report['environment']['status'] == 'violated'
        assert env.report['workload']['status'] == 'succeeded'


def test_counter_reset_is_a_violation():
    env = Environment(probes=[Probe([5,5,4])],rules=[MetricRule('test','temperature',max_increase=0)],
                      sample_interval_s=10)
    env.run(lambda: None)
    assert any(v['reason']=='counter_decreased' for v in env.report['environment']['violations'])


def test_synchronization_precedes_restore(tmp_path):
    events = []
    env = Environment(controls=[Control('a',events)],output_dir=tmp_path/'run',lock=Lock(),
                      synchronize=lambda: events.append('sync'))
    env.run(lambda: events.append('work'))
    assert events[-3:] == ['work','sync','restore:a']


def test_output_never_overwrites(tmp_path):
    path = tmp_path/'run'
    Environment(output_dir=path).run(lambda: None)
    with pytest.raises(FileExistsError):
        Environment(output_dir=path).run(lambda: None)


def test_single_use_and_decorator():
    env = Environment()
    assert env.run(lambda: 3) == 3
    with pytest.raises(RuntimeError):
        env.run(lambda: 4)
    @controlled(lambda: Environment())
    def twice(value):
        return value * 2
    assert twice(2) == 4 and twice(3) == 6


@pytest.mark.parametrize('kwargs', [dict(sample_interval_s=True),dict(ready_timeout_s=float('nan')),
                                    dict(stable_for_s=-1),dict(warmup_runs=True),dict(on_violation='ignore')])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Environment(**kwargs)


def test_rules_reject_invalid_values():
    for kwargs in (dict(maximum=True),dict(max_increase=-1),dict(minimum=8,maximum=2)):
        with pytest.raises(ValueError):
            MetricRule('test','value',**kwargs)


def test_monitor_collects_during_workload():
    env = Environment(probes=[Probe([40])],sample_interval_s=.005)
    env.run(lambda: time.sleep(.025))
    assert env.report['monitor']['samples'] >= 3


def test_baseline_must_remain_inside_stable_temperature_window():
    stamp = [0]
    def advance(seconds):
        stamp[0] += int(seconds * 1e9)
    env = Environment(probes=[Probe([40,40,80])],rules=[MetricRule('test','temperature',max_span=1)],
                      ready_timeout_s=.2,stable_for_s=.1,sample_interval_s=.1,
                      clock_ns=lambda:stamp[0],sleep=advance)
    with pytest.raises(ReadinessError):
        env.run(lambda: pytest.fail('baseline jumped outside stable window'))
    assert env.report['workload']['status'] == 'not_started'
