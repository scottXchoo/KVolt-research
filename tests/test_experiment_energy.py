"""Generic experiment energy contracts, independent of models and CUDA."""
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import time

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.telemetry import (
    EnergyMeter, PowerProfile, RAILS, SCOPE, discover_power_profile,
    interval_energy_metrics,
)

S = 10**9


def rows():
    # Three independent P(t)=2+t W curves; integral 1..3 is 8 J each.
    return [dict(rail=rail, timestamp_ns=t*S, power_w=2+t, error=None)
            for rail in RAILS for t in (0, 2, 4)]


def make_hwmon(root, *, name='ina3221', labels=RAILS):
    sensor = root/'hwmon7'
    sensor.mkdir(parents=True)
    (sensor/'name').write_text(name+'\n')
    for index, label in enumerate(labels, 1):
        (sensor/f'in{index}_label').write_text(label+'\n')
        (sensor/f'in{index}_input').write_text('5000\n')
        (sensor/f'curr{index}_input').write_text('100\n')
    return sensor


@pytest.fixture
def profile(tmp_path):
    make_hwmon(tmp_path)
    return discover_power_profile(tmp_path)


def test_interval_analytical_energy_and_power():
    result = interval_energy_metrics(rows(), S, 3*S, max_gap_ns=2*S)
    assert result['valid'], result
    assert result['duration_s'] == 2
    assert result['energy_j'] == pytest.approx(24)
    assert result['average_power_w'] == pytest.approx(12)
    assert result['scope'] == SCOPE
    assert result['sensor_response_validated'] is False
    for rail in RAILS:
        assert result['rails'][rail]['energy_j'] == pytest.approx(8)
        assert result['rails'][rail]['average_power_w'] == pytest.approx(4)
    signed = [dict(row, power_w=-1) for row in rows()]
    assert interval_energy_metrics(signed, S, 3*S, max_gap_ns=2*S)['energy_j'] == -6


@pytest.mark.parametrize('mode', ['nan', 'error', 'missing', 'duplicate', 'left', 'right', 'gap', 'outside'])
def test_interval_rejects_invalid_trace_without_zero_filling(mode):
    samples = rows()
    gap = 2*S
    if mode == 'nan':
        samples[1]['power_w'] = math.nan
    elif mode == 'error':
        samples[1]['error'] = {'type': 'OSError'}
    elif mode == 'missing':
        samples = [row for row in samples if row['rail'] != RAILS[1]]
    elif mode == 'duplicate':
        samples[1]['timestamp_ns'] = 0
    elif mode == 'left':
        samples[0]['timestamp_ns'] = S
    elif mode == 'right':
        samples[2]['timestamp_ns'] = 3*S
    elif mode == 'gap':
        gap -= 1
    else:
        samples.insert(0, dict(rail=RAILS[0], timestamp_ns=0, power_w=None,
                               error={'type': 'OSError'}))
    result = interval_energy_metrics(samples, S, 3*S, max_gap_ns=gap)
    assert not result['valid'] and result['reason']
    assert result['energy_j'] is None and result['average_power_w'] is None


@pytest.mark.parametrize('start,end', [(S, S), (2*S, S), (True, S), (-1, S), (0.0, S)])
def test_interval_rejects_boundaries(start, end):
    result = interval_energy_metrics(rows(), start, end, max_gap_ns=2*S)
    assert not result['valid'] and result['energy_j'] is None


def test_discovery_uses_labels_instead_of_hwmon_number(tmp_path):
    sensor = make_hwmon(tmp_path, labels=(RAILS[2], RAILS[0], RAILS[1]))
    other = tmp_path/'hwmon0'; other.mkdir()
    (other/'name').write_text('unrelated\n')
    profile = discover_power_profile(tmp_path)
    assert isinstance(profile, PowerProfile)
    assert Path(profile.rails[RAILS[0]].current_path) == sensor/'curr2_input'
    assert profile.interval_ns == 20_000_000 and profile.max_gap_ns == 200_000_000
    assert profile.ready_timeout_ns == 5*S and profile.coverage_timeout_ns == 2*S


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'bad_voltage', 'no_device'])
def test_discovery_rejects_incomplete_ambiguous_or_unreadable_sensor(tmp_path, mode):
    sensor = make_hwmon(tmp_path)
    if mode == 'missing':
        (sensor/'in3_label').unlink()
    elif mode == 'duplicate':
        second = tmp_path/'hwmon8'; second.mkdir()
        (second/'name').write_text('ina3221')
        (second/'in1_label').write_text(RAILS[0])
        (second/'in1_input').write_text('5000')
        (second/'curr1_input').write_text('100')
    elif mode == 'bad_voltage':
        (sensor/'in2_input').unlink()
    else:
        (sensor/'name').write_text('another-device')
    with pytest.raises(ContractError):
        discover_power_profile(tmp_path)


def test_discovery_ignores_nonphysical_summation_channel(tmp_path):
    sensor = make_hwmon(tmp_path)
    (sensor/'in4_label').write_text(RAILS[0])
    profile = discover_power_profile(tmp_path)
    assert len(profile.rails) == 3


def test_bad_numeric_sensor_is_rejected_by_bounded_collector_start(tmp_path):
    sensor = make_hwmon(tmp_path)
    (sensor/'curr2_input').write_text('not-a-number')
    profile = replace(discover_power_profile(tmp_path), ready_timeout_ns=500_000_000)
    meter = EnergyMeter(profile)
    with pytest.raises(ContractError, match='not ready'):
        meter.start()
    assert not meter.result['valid'] and meter.result['energy_j'] is None
    assert any(row['error'] for row in meter.receipt['samples'])


def fake_meter(profile, *, events=None, receipt_errors=None, sync=None,
               start_receipt=None, start_error=None, finish_error=None, stop_error=None):
    events = [] if events is None else events
    profile = replace(profile, max_gap_ns=2*S)

    class Collector:
        def __init__(self, profile, **kwargs):
            self.profile = profile
            self.receipt = dict(samples=rows(), raw_trace_jsonl='{"source":"fake"}\n',
                errors=receipt_errors or [], profile=profile.to_dict(), status='stopped',
                ready=True, scope=SCOPE, sensor_response_validated=False)

        def start(self):
            events.append('collector.start')
            if start_error is not None:
                raise start_error
            return {'ready': True, 'errors': []} if start_receipt is None else start_receipt

        def finish(self, end_ns):
            events.append(('collector.finish', end_ns))
            if finish_error is not None:
                raise finish_error
            return self.receipt

        def stop(self):
            events.append('collector.stop')
            if stop_error is not None:
                raise stop_error
            return self.receipt

    times = iter((S, 3*S))

    def clock():
        value = next(times)
        events.append(('clock', value))
        return value

    meter = EnergyMeter(profile, clock=clock, collector_factory=Collector,
                        synchronize=sync)
    return meter, events


def test_manual_lifecycle_synchronizes_before_boundaries_and_is_idempotent(profile):
    events = []
    meter, _ = fake_meter(profile, events=events, sync=lambda: events.append('sync'))
    assert meter.result is None and meter.start() is meter
    events.append('workload')
    result = meter.stop()
    assert result['valid'] and result['energy_j'] == 24
    assert (result['start_ns'], result['end_ns'], result['completed']) == (S, 3*S, True)
    assert events[:7] == ['collector.start', 'sync', ('clock', S), 'workload',
                          'sync', ('clock', 3*S), ('collector.finish', 3*S)]
    before = list(events)
    assert meter.stop() is result and events == before
    assert meter.receipt['samples'] == rows()
    assert 'samples' not in result['collector'] and 'raw_trace_jsonl' not in result['collector']
    with pytest.raises(ContractError):
        meter.start()


def test_context_manager_preserves_original_workload_exception(profile):
    meter, events = fake_meter(profile)
    original = LookupError('workload failure')
    with pytest.raises(LookupError) as caught:
        with meter:
            raise original
    assert caught.value is original
    assert not meter.result['valid'] and meter.result['reason'] == 'workload_failed'
    assert meter.result['energy_j'] is None and not meter.result['completed']
    assert meter.receipt is not None


def test_disabling_active_meter_still_finishes_and_cleans_up_collector(profile):
    meter, events = fake_meter(profile)
    meter.start()
    meter.enabled = False
    result = meter.stop()
    assert result['valid'] and result['energy_j'] == 24
    assert ('collector.finish', 3*S) in events and 'collector.stop' in events


def test_stop_before_start_is_contract_error(profile):
    meter, events = fake_meter(profile)
    with pytest.raises(ContractError):
        meter.stop()
    assert not events


@pytest.mark.parametrize('mode', ['not_ready', 'start_error'])
def test_startup_failure_stops_collector_and_retains_diagnostics(profile, mode):
    original = RuntimeError('collector startup failed')
    kwargs = ({'start_receipt': {'ready': False, 'errors': ['unavailable']}}
              if mode == 'not_ready' else {'start_error': original})
    meter, events = fake_meter(profile, **kwargs)
    with pytest.raises(ContractError if mode == 'not_ready' else RuntimeError) as caught:
        meter.start()
    if mode == 'start_error':
        assert caught.value is original
    assert 'collector.stop' in events and meter.receipt is not None
    assert not meter.result['valid'] and meter.result['energy_j'] is None


def test_finish_and_stop_errors_coexist_without_fabricated_energy(profile):
    meter, events = fake_meter(profile, finish_error=RuntimeError('finish failed'),
                               stop_error=OSError('stop failed'))
    meter.start()
    result = meter.stop()
    assert not result['valid'] and result['energy_j'] is None
    assert {error['stage'] for error in result['errors']} >= {'collector_finish', 'collector_stop'}
    assert 'collector.stop' in events


@pytest.mark.parametrize('stage', ['finish', 'stop'])
def test_collector_keyboard_interrupt_is_preserved_after_cleanup(profile, stage):
    original = KeyboardInterrupt('user interrupt')
    meter, events = fake_meter(profile, **{stage+'_error': original})
    meter.start()
    with pytest.raises(KeyboardInterrupt) as caught:
        meter.stop()
    assert caught.value is original and 'collector.stop' in events
    assert meter.result['errors'] and meter.result['energy_j'] is None


def test_workload_exception_survives_collector_interrupt(profile):
    meter, events = fake_meter(profile, finish_error=KeyboardInterrupt('cleanup interrupt'))
    original = LookupError('workload failed first')
    with pytest.raises(LookupError) as caught:
        with meter:
            raise original
    assert caught.value is original and 'collector.stop' in events
    assert meter.result['reason'] == 'workload_failed' and meter.result['errors']


def test_disabled_path_calls_no_sensor_worker_sync_or_clock(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled path touched measurement')
    import kvolt_runtime.telemetry.experiment as module
    monkeypatch.setattr(module, 'discover_power_profile', forbidden)
    with EnergyMeter(enabled=False, synchronize=forbidden, clock=forbidden,
                     collector_factory=forbidden) as meter:
        pass
    assert not meter.result['valid'] and meter.result['reason'] == 'not_collected'
    assert meter.result['energy_j'] is None and meter.result['duration_s'] is None
    meter.save(tmp_path/'disabled')


def test_collector_errors_invalidate_otherwise_good_energy(profile):
    meter, _ = fake_meter(profile, receipt_errors=[{'code': 'COLLECTOR_RIGHT_COVERAGE'}])
    meter.start()
    result = meter.stop()
    assert not result['valid'] and result['energy_j'] is None and result['errors']


@pytest.mark.parametrize('fail_on', [1, 2])
def test_sync_failure_cleans_up_and_preserves_exception(profile, fail_on):
    calls = []
    original = RuntimeError('sync failure')

    def synchronize():
        calls.append(1)
        if len(calls) == fail_on:
            raise original

    meter, events = fake_meter(profile, sync=synchronize)
    with pytest.raises(RuntimeError) as caught:
        if fail_on == 1:
            meter.start()
        else:
            meter.start(); meter.stop()
    assert caught.value is original and 'collector.stop' in events
    assert meter.receipt is not None


def test_workload_exception_survives_cleanup_sync_failure(profile):
    count = []

    def synchronize():
        count.append(1)
        if len(count) == 2:
            raise RuntimeError('cleanup sync failed')

    meter, _ = fake_meter(profile, sync=synchronize)
    with pytest.raises(KeyError, match='original'):
        with meter:
            raise KeyError('original')
    assert meter.result['errors'] and meter.result['energy_j'] is None


def test_save_preserves_raw_trace_csv_and_rejects_overwrite(profile, tmp_path):
    meter, _ = fake_meter(profile)
    with pytest.raises(ContractError):
        meter.save(tmp_path/'unfinished')
    with meter:
        pass
    output = tmp_path/'measurement'
    meter.save(output)
    saved = json.loads((output/'energy.json').read_text())
    assert saved['energy_j'] == 24 and saved['scope'] == SCOPE
    raw = (output/'power-trace.jsonl').read_bytes()
    assert raw == meter.receipt['raw_trace_jsonl'].encode()
    assert (output/'power-samples.csv').read_text().startswith('rail,timestamp_ns,')
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    with pytest.raises((ContractError, FileExistsError)):
        meter.save(output)
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}


def test_actual_spawn_fake_sysfs_measures_without_cuda(profile):
    with EnergyMeter(profile) as meter:
        time.sleep(.06)
    result = meter.result
    assert result['valid'], result
    assert result['average_power_w'] == pytest.approx(1.5)
    assert result['energy_j'] == pytest.approx(1.5*result['duration_s'])
    assert result['collector']['start_method'] == 'spawn'
