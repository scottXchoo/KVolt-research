"""Request memory evidence must survive storage without entering token timing."""
import json
from types import SimpleNamespace

import pytest

import kvolt_runtime.execution.runner as runner_module
from kvolt_runtime.telemetry.memory import RequestMemoryMonitor, no_memory
from test_session_runner import setup


class Cuda:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.allocated, self.reserved = 100, 200
        self.peak_allocated, self.peak_reserved = 9000, 10000

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def read(self, name, device):
        assert device == 'cuda:0'
        self.calls.append((name, self.clock.now))
        return getattr(self, name)

    def reset_peak_memory_stats(self, device):
        self.read('allocated', device)
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved
        self.calls.append(('reset', self.clock.now))

    def memory_allocated(self, device):
        return self.read('allocated', device)

    def memory_reserved(self, device):
        return self.read('reserved', device)

    def max_memory_allocated(self, device):
        return self.read('peak_allocated', device)

    def max_memory_reserved(self, device):
        return self.read('peak_reserved', device)


def proc_fixture(tmp_path):
    root = tmp_path / 'proc'
    (root / 'self').mkdir(parents=True)
    (root / 'meminfo').write_text(
        'MemTotal: 1000 kB\nMemAvailable: 250 kB\nSwapTotal: 100 kB\nSwapFree: 80 kB\n')
    (root / 'self/status').write_text('VmRSS: 30 kB\nVmHWM: 50 kB\nVmSwap: 5 kB\n')
    return root


@pytest.mark.parametrize('nmax', [1, 3])
def test_request_memory_outside_timing_and_persisted_with_isolated_peaks(tmp_path, monkeypatch, nmax):
    runner, prepared, options, backend, _, emitted, _ = setup(tmp_path, requests=2, nmax=nmax)
    proc_root = proc_fixture(tmp_path)
    cuda = Cuda(runner.clock)
    backend.torch = SimpleNamespace(cuda=cuda)
    monkeypatch.setattr(runner_module, 'RequestMemoryMonitor',
                        lambda **kwargs: RequestMemoryMonitor(proc_root=proc_root, **kwargs))
    prefill, release = backend.prefill_raw, backend.release_request

    def allocate(request):
        cuda.peak_allocated, cuda.peak_reserved = ((1200, 1600) if request.request_id.endswith('0')
                                                  else (450, 800))
        cuda.allocated, cuda.reserved = cuda.peak_allocated, cuda.peak_reserved
        return prefill(request)

    def free(cache):
        release(cache)
        cuda.allocated, cuda.reserved = 100, 200

    backend.prefill_raw, backend.release_request = allocate, free
    session = runner.run(prepared, options)
    assert session['exit_code'] == 0
    assert len([call for call in cuda.calls if call[0] == 'reset']) == 2
    for index, result in enumerate(session['results']):
        memory, timing = result['memory'], result['timing']
        assert memory['status'] == 'valid' and memory['generation_status'] == 'completed'
        assert memory['before']['read_end_ns'] < timing['t0_ns']
        assert memory['cuda_peak_reset']['read_end_ns'] < timing['t0_ns']
        assert timing['tN_ns'] < memory['after']['read_start_ns']
        assert memory['after']['read_end_ns'] < result['cleanup_started_ns']
        assert memory['after']['system']['usage_percent'] == 75.0
        assert memory['after']['process']['rss_bytes'] == 30 * 1024
        assert memory['after']['cuda']['max_allocated_bytes'] == [1200, 450][index]
        assert memory['after']['cuda']['max_reserved_bytes'] == [1600, 800][index]
        assert memory['after']['cuda']['allocated_bytes'] == [1200, 450][index]
        assert all(not timing['t0_ns'] <= time <= timing['tN_ns'] for _, time in cuda.calls)
        assert (timing['mean_tpot_s'] is None) == (nmax == 1)
        saved = json.loads((tmp_path / f'out/requests/{index:06d}/result.json').read_text())
        assert saved['memory'] == memory
        assert emitted[index]['memory_status'] == memory['status']
    saved_session = json.loads((tmp_path / 'out/session.json').read_text())
    assert [row['memory'] for row in saved_session['results']] == [row['memory'] for row in session['results']]


@pytest.mark.parametrize('failure', ['constructor', 'start', 'finish'])
def test_auxiliary_memory_error_does_not_invalidate_generation_or_energy(tmp_path, monkeypatch, failure):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, collect=True)

    class BrokenMonitor:
        def __init__(self, **kwargs):
            if failure == 'constructor':
                raise RuntimeError('memory unavailable')

        def start(self):
            if failure == 'start':
                raise RuntimeError('memory unavailable')

        def finish(self):
            raise RuntimeError('memory unavailable')

    monkeypatch.setattr(runner_module, 'RequestMemoryMonitor', BrokenMonitor)
    session = runner.run(prepared, options)
    result = session['results'][0]
    assert session['exit_code'] == 0 and result['errors'] == []
    assert result['generation']['status'] == 'completed'
    assert result['timing']['valid'] and result['energy']['valid']
    assert result['cleanup']['status'] == 'completed'
    assert result['memory']['reason'] == 'collection_failed'
    assert result['memory']['before'] is None and result['memory']['after'] is None
    expected_stage = 'memory_finish' if failure == 'finish' else 'memory_start'
    assert result['memory']['errors'][0]['stage'] == expected_stage


def test_unavailable_memory_does_not_abort_later_requests(tmp_path, monkeypatch):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, requests=2)
    monkeypatch.setattr(runner_module, 'RequestMemoryMonitor',
                        lambda **kwargs: RequestMemoryMonitor(proc_root=tmp_path / 'missing', **kwargs))
    session = runner.run(prepared, options)
    assert session['exit_code'] == 0
    assert session['counts']['generation_completed'] == 2
    for result in session['results']:
        assert result['memory']['status'] == 'unavailable'
        assert result['memory']['after']['system']['usage_percent'] is None
        assert result['memory']['after']['process']['rss_bytes'] is None
        assert result['memory']['after']['cuda']['max_allocated_bytes'] is None


@pytest.mark.parametrize('failure', ['prefill', 'pack', 'decode'])
def test_failed_request_retains_memory_but_not_run_has_no_measurements(tmp_path, monkeypatch, failure):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, requests=2, fail=failure)
    proc_root = proc_fixture(tmp_path)
    monkeypatch.setattr(runner_module, 'RequestMemoryMonitor',
                        lambda **kwargs: RequestMemoryMonitor(proc_root=proc_root, **kwargs))
    session = runner.run(prepared, options)
    failed, unattempted = session['results']
    assert session['exit_code'] == 1
    assert failed['memory']['generation_status'] == 'failed'
    assert failed['memory']['before']['process']['rss_bytes'] == 30 * 1024
    assert failed['memory']['after']['process']['rss_bytes'] == 30 * 1024
    assert unattempted['memory']['status'] == 'not_run'
    assert unattempted['memory']['before'] is None and unattempted['memory']['after'] is None


def test_slow_memory_snapshots_leave_token_timing_unchanged(tmp_path, monkeypatch):
    class SlowMonitor:
        def __init__(self, *, clock, **kwargs):
            self.clock = clock

        def start(self):
            self.clock.now += 10**12

        def finish(self):
            self.clock.now += 10**12
            return no_memory('test_only')

    baseline_root, slow_root = tmp_path / 'baseline', tmp_path / 'slow'
    baseline_root.mkdir()
    slow_root.mkdir()
    baseline, inputs, options, _, _, _, _ = setup(baseline_root)
    expected = baseline.run(inputs, options)['results'][0]
    monkeypatch.setattr(runner_module, 'RequestMemoryMonitor', SlowMonitor)
    runner, inputs, options, _, _, _, _ = setup(slow_root)
    actual = runner.run(inputs, options)['results'][0]
    assert actual['timing']['valid']
    assert actual['timing']['ttft_s'] == expected['timing']['ttft_s']
    assert actual['timing']['mean_tpot_s'] == expected['timing']['mean_tpot_s']
    assert actual['generation']['token_ids'] == expected['generation']['token_ids']


@pytest.mark.parametrize('stop_at', ['prefill', 'decode'])
def test_interrupted_generation_keeps_memory_with_its_generation_status(tmp_path, monkeypatch, stop_at):
    runner, prepared, options, _, _, _, _ = setup(tmp_path, stop_at=stop_at)
    proc_root = proc_fixture(tmp_path)
    monkeypatch.setattr(runner_module, 'RequestMemoryMonitor',
                        lambda **kwargs: RequestMemoryMonitor(proc_root=proc_root, **kwargs))
    result = runner.run(prepared, options)['results'][0]
    assert result['memory']['generation_status'] == 'interrupted'
    assert result['memory']['after']['process']['rss_bytes'] == 30 * 1024
    assert not result['timing']['valid']
    assert result['cleanup']['status'] == 'completed'
