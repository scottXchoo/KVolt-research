"""Real CPU controllers consume prior completed request observations only."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from kvolt_runtime.execution.runner import SessionRunner
from kvolt_runtime.policy.decode import DecodeController
from kvolt_runtime.policy.online import OnlineLUTRepository
from test_session_runner import Adapter, Collector, setup
from test_run_cli import arguments, factories


def prefill_cell(repo, frequency):
    return next(p for p in repo.prefill(1024) if p.frequency_mhz == frequency)


def decode_cell(repo, group, frequency):
    return next(p for p in repo.decode(1024) if p.group == group and p.frequency_mhz == frequency)


def online_setup(tmp_path, *, alpha=0.1, **kwargs):
    runner, prepared, options, backend, events, emitted, collector = setup(tmp_path, **kwargs)
    repo = OnlineLUTRepository(runner.repo, state_path=tmp_path / 'state.json',
                               context={'test_runtime': True}, alpha=alpha)
    # A fresh trace for each request, as in the real CLI.
    runner = SessionRunner(backend=backend, repo=repo, device=runner.device,
        recorder=runner.recorder, decode_controller=DecodeController(repo, Adapter(events), clock=runner.clock),
        clock=runner.clock, stop=runner.stop, emit=emitted.append, warmup=runner.warmup,
        collector_factory=lambda index: Collector(runner.clock, events,
                                                  failure=kwargs.get('collector_failure')))
    return runner, prepared, options, backend, events, emitted, repo


def test_request_feedback_changes_next_selection_and_records_reproducible_states(tmp_path):
    runner, prepared, options, backend, events, emitted, repo = online_setup(
        tmp_path, requests=2, collect=True, alpha=1.0)
    prefill = backend.prefill_raw
    def delayed(request):
        backend.clock.now += 2_000_000_000
        return prefill(request)
    backend.prefill_raw = delayed
    session = runner.run(prepared, options, session_id='online-test')
    first, second = session['results']
    assert session['exit_code'] == 0
    assert first['slo']['ttft_met'] is False  # Valid SLO misses remain observations.
    assert first['online_lut']['used']['revision'] == 0
    assert second['online_lut']['used']['revision'] == first['online_lut']['feedback']['after_revision'] == 1
    assert first['decisions']['prefill']['selected']['frequency_mhz'] == 510
    assert second['decisions']['prefill']['selected']['frequency_mhz'] == 612
    assert repo.snapshot()['revision'] == 2
    assert first['energy']['valid'] and second['energy']['valid']
    selected = first['decisions']['decode']['selected']
    assert decode_cell(repo, selected['group'], selected['frequency_mhz']).tpot_s == second['timing']['mean_tpot_s']
    assert prefill_cell(repo, 510).ttft_s == first['timing']['ttft_s']
    root = Path(options.output_dir)
    assert json.loads((root / 'online-lut.initial.json').read_text())['revision'] == 0
    assert json.loads((root / 'online-lut.final.json').read_text())['revision'] == 2
    saved = json.loads((root / 'requests/000000/result.json').read_text())
    assert saved['online_lut']['feedback']['after_revision'] == 1
    raw = json.loads((root / 'requests/000000/online-observation.json').read_text())
    assert raw['online_lut']['observation_id'] == 'online-test:0'
    assert 'feedback' not in raw['online_lut']
    assert emitted[0]['online_lut']['after_revision'] == 1
    restarted = OnlineLUTRepository(repo.base, state_path=tmp_path / 'state.json',
                                    context={'test_runtime': True}, alpha=1.0)
    assert restarted.snapshot()['revision'] == 2
    assert prefill_cell(restarted, 510) == prefill_cell(repo, 510)


@pytest.mark.parametrize('kwargs,expected_updates', [({}, True), ({'nmax': 1}, True),
    ({'fail': 'decode'}, False), ({'device_fail': 'restore'}, True),
    ({'collect': True, 'collector_failure': 'finish'}, True)])
def test_feedback_measurement_eligibility(tmp_path, kwargs, expected_updates):
    runner, prepared, options, _, _, _, repo = online_setup(tmp_path, **kwargs)
    session = runner.run(prepared, options)
    result = session['results'][0]
    feedback = result['online_lut']['feedback']
    assert bool(feedback['updated']) is expected_updates
    if kwargs.get('nmax') == 1:
        assert result['timing']['mean_tpot_s'] is None
        assert decode_cell(repo, 128, 408) == decode_cell(repo.base, 128, 408)
    if not kwargs.get('collect'):
        assert result['energy']['status'] == 'not_collected'
        frequency = result['decisions']['prefill']['selected']['frequency_mhz']
        assert prefill_cell(repo, frequency).energy_j == prefill_cell(repo.base, frequency).energy_j


def test_feedback_write_failure_retains_measurement_and_stops_next_request(tmp_path, monkeypatch):
    runner, prepared, options, _, _, emitted, repo = online_setup(tmp_path, requests=2)
    def fail(*args, **kwargs):
        raw = Path(options.output_dir) / 'requests/000000/online-observation.json'
        assert raw.is_file()  # Durable evidence must precede state mutation.
        raise OSError('state fsync failed')
    monkeypatch.setattr(repo, 'observe', fail)
    session = runner.run(prepared, options)
    first, second = session['results']
    assert session['exit_code'] == 1
    assert first['timing']['valid'] and first['timing']['ttft_s'] > 0
    assert first['online_lut']['feedback']['status'] == 'failed'
    assert any(error['stage'] == 'online_lut_feedback' for error in first['errors'])
    assert second['generation']['status'] == 'not_run'
    saved = json.loads((Path(options.output_dir) / 'requests/000000/result.json').read_text())
    assert saved['timing'] == first['timing']
    assert emitted[0]['online_lut']['status'] == 'failed'
    assert repo.snapshot()['revision'] == 0


def test_adaptive_performance_rejected_before_hardware(tmp_path):
    runner, prepared, options, _, events, _, _ = online_setup(tmp_path)
    session = runner.run(prepared, replace(options, mode='performance', collect_power=True))
    assert session['exit_code'] == 2
    assert events == []


def test_cli_online_default_frozen_and_explicit_reuse(tmp_path):
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events)
    seen = []
    original = seam.decode_controller
    def decode(repo, *args):
        seen.append(repo)
        return original(repo, *args)
    assert main(arguments(tmp_path), factories=replace(seam, decode_controller=decode), emit=lambda _: None) == 0
    assert isinstance(seen[0], OnlineLUTRepository)
    saved = json.loads((tmp_path / 'out/session.json').read_text())
    assert saved['provenance']['lut']['kind'] == 'synthetic'
    assert saved['provenance']['online_lut']['enabled'] is True
    assert seen[0].snapshot()['state_path'] == str(tmp_path / 'out/online-lut.json')
    (tmp_path / 'other').mkdir()
    events.clear()
    assert main(arguments(tmp_path / 'other', '--no-online-lut'), factories=replace(seam, decode_controller=decode), emit=lambda _: None) == 0
    assert not isinstance(seen[1], OnlineLUTRepository)


@pytest.mark.parametrize('flags', [ ['--no-online-lut', '--online-lut-alpha', '0.1'],
    ['--no-online-lut', '--online-lut-state', 'state.json'], ['--online-lut-alpha', '0'],
    ['--online-lut-alpha', 'nan'], ['--online-lut-alpha', '1.1'],
    ['--mode', 'performance', '--online-lut-alpha', '0.1'],
    ['--mode', 'performance', '--online-lut-state', 'state.json'] ])
def test_cli_online_conflicts_rejected_before_source_or_hardware(tmp_path, flags):
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events)
    assert main(arguments(tmp_path, *flags), factories=seam, emit=lambda _: None) == 2
    assert events == []


def test_original_observation_storage_failure_prevents_feedback(tmp_path, monkeypatch):
    runner, prepared, options, _, _, _, repo = online_setup(tmp_path, requests=2)
    original = runner.recorder.atomic_json
    def fail(path, value):
        if path.endswith('/online-observation.json'):
            raise OSError('observation fsync failed')
        return original(path, value)
    monkeypatch.setattr(runner.recorder, 'atomic_json', fail)
    session = runner.run(prepared, options)
    assert session['exit_code'] == 1
    assert repo.snapshot()['revision'] == 0
    assert session['results'][0]['timing']['valid']
    assert session['results'][1]['generation']['status'] == 'not_run'
    saved = json.loads((Path(options.output_dir) / 'requests/000000/result.json').read_text())
    assert saved['online_lut']['feedback']['status'] == 'failed'
    assert saved['online_lut']['observation_path'] is None


def test_initial_state_storage_precedes_hardware(tmp_path, monkeypatch):
    runner, prepared, options, _, events, _, _ = online_setup(tmp_path)
    original = runner.recorder.atomic_json
    def fail(path, value):
        if path == 'online-lut.initial.json':
            assert events == []
            raise OSError('initial fsync failed')
        return original(path, value)
    monkeypatch.setattr(runner.recorder, 'atomic_json', fail)
    session = runner.run(prepared, options)
    assert session['exit_code'] == 1 and events == []
    assert session['results'][0]['generation']['status'] == 'not_run'


def test_cli_reuses_relocated_state_and_rejects_changed_power_timing_before_hardware(tmp_path):
    from kvolt_runtime.run_cli import main
    events, seen = [], []
    seam, _ = factories(events)
    def decode(repo, *args):
        seen.append(repo)
        return object()
    seam = replace(seam, decode_controller=decode,
        device_profile=lambda path: type('Profile', (), {'as_dict': lambda _: {'timeout_ns': 17}})())
    assert main(arguments(tmp_path), factories=seam, emit=lambda _: None) == 0
    shared = tmp_path / 'shared.json'
    shared.write_text(json.dumps(seen[0].export_state()))
    second = tmp_path / 'second'
    second.mkdir()
    events.clear()
    assert main(arguments(second, '--online-lut-state', str(shared)), factories=seam, emit=lambda _: None) == 0
    assert seen[0].snapshot()['binding_sha256'] == seen[1].snapshot()['binding_sha256']
    third = tmp_path / 'third'
    third.mkdir()
    events.clear()
    changed = replace(seam,
        device_profile=lambda path: type('Profile', (), {'as_dict': lambda _: {'timeout_ns': 18}})())
    assert main(arguments(third, '--online-lut-state', str(shared)), factories=changed, emit=lambda _: None) == 2
    assert 'lock' not in events and 'backend' not in events
    saved = json.loads((third / 'out/session.json').read_text())
    assert saved['errors'][0]['stage'] == 'online_lut'


def test_cli_initial_state_failure_prevents_hardware(tmp_path):
    from kvolt_runtime.recording import SessionRecorder
    from kvolt_runtime.run_cli import main
    events = []
    seam, _ = factories(events)
    class Recorder(SessionRecorder):
        def atomic_json(self, path, value):
            if path == 'online-lut.initial.json':
                raise OSError('initial state write failed')
            return super().atomic_json(path, value)
    assert main(arguments(tmp_path), factories=replace(seam, recorder=Recorder), emit=lambda _: None) == 1
    assert 'lock' not in events and 'backend' not in events


def test_post_commit_failure_reports_actual_revision_and_aborts(tmp_path, monkeypatch):
    runner, prepared, options, _, _, _, repo = online_setup(tmp_path, requests=2)
    observe = repo.observe
    def fail_after_commit(*args, **kwargs):
        observe(*args, **kwargs)
        raise OSError('directory fsync uncertain after replace')
    monkeypatch.setattr(repo, 'observe', fail_after_commit)
    session = runner.run(prepared, options)
    first, second = session['results']
    assert session['exit_code'] == 1
    assert first['online_lut']['feedback']['status'] == 'failed'
    assert first['online_lut']['feedback']['after_revision'] == 1
    assert first['online_lut']['feedback']['state']['revision'] == 1
    assert second['generation']['status'] == 'not_run'
    assert session['online_lut']['final']['revision'] == 1
    assert first['timing']['valid']


def test_feedback_work_is_outside_timed_intervals(tmp_path, monkeypatch):
    runner, prepared, options, _, _, _, repo = online_setup(tmp_path, requests=2)
    observe = repo.observe
    measured = []
    def expensive_feedback(result, **kwargs):
        assert runner.clock.now >= result['cleanup_ended_ns']
        measured.append(dict(result['timing']))
        runner.clock.now += 30_000_000_000
        return observe(result, **kwargs)
    monkeypatch.setattr(repo, 'observe', expensive_feedback)
    session = runner.run(prepared, options)
    assert session['exit_code'] == 0
    assert [result['timing'] for result in session['results']] == measured
    assert session['results'][0]['timing']['ttft_s'] == session['results'][1]['timing']['ttft_s']
    assert session['results'][1]['timing']['t0_ns'] - session['results'][0]['timing']['tN_ns'] > 30_000_000_000


def test_cli_changed_power_sensor_path_rejected_before_hardware(tmp_path):
    from kvolt_runtime.run_cli import main
    events, seen = [], []
    seam, _ = factories(events)
    def decode(repo, *args):
        seen.append(repo)
        return object()
    def power_profile(sensor):
        return lambda path: type('Power', (), {'to_dict': lambda _: {
            'rails': {'VDD_GPU_SOC': {'voltage_path': '/sys/voltage', 'current_path': sensor}},
            'max_gap_ns': 20_000_000}})()
    seam = replace(seam, decode_controller=decode, power_profile=power_profile('/sys/current-a'))
    first_args = arguments(tmp_path, '--collect-power', '--power-profile', str(tmp_path / 'power.json'))
    assert main(first_args, factories=seam, emit=lambda _: None) == 0
    shared = tmp_path / 'shared.json'
    shared.write_text(json.dumps(seen[0].export_state()))
    second = tmp_path / 'second'
    second.mkdir()
    events.clear()
    changed = replace(seam, power_profile=power_profile('/sys/current-b'))
    second_args = arguments(second, '--collect-power', '--power-profile', str(second / 'power.json'),
                            '--online-lut-state', str(shared))
    assert main(second_args, factories=changed, emit=lambda _: None) == 2
    assert 'lock' not in events and 'backend' not in events
