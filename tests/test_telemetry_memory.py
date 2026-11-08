"""CPU-only contract tests for request memory diagnostics."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvolt_runtime.telemetry.memory import RequestMemoryMonitor, no_memory


MEMINFO = 'MemTotal: 1000 kB\nMemAvailable: 250 kB\nSwapTotal: 200 kB\nSwapFree: 125 kB\n'
STATUS = 'Name: kvolt\nVmRSS: 100 kB\nVmHWM: 150 kB\nVmSwap: 10 kB\n'


def proc_fixture(root, meminfo=MEMINFO, status=STATUS):
    root.mkdir(exist_ok=True)
    (root / 'self').mkdir(exist_ok=True)
    (root / 'meminfo').write_text(meminfo)
    (root / 'self' / 'status').write_text(status)
    return root


class Cuda:
    def __init__(self):
        self.calls = []
        self.current = [100, 200]
        self.peaks = [9000, 10000]  # A previous request/warmup peak.
        self.initialized = True
        self.available = True
        self.fail = set()

    def is_initialized(self):
        self.calls.append(('is_initialized', None))
        return self.initialized

    def is_available(self):
        self.calls.append(('is_available', None))
        return self.available

    def _call(self, name, device, value):
        self.calls.append((name, device))
        if name in self.fail:
            raise RuntimeError(name + ' failed')
        return value

    def reset_peak_memory_stats(self, device):
        self._call('reset_peak_memory_stats', device, None)
        self.peaks = self.current.copy()

    def memory_allocated(self, device):
        return self._call('memory_allocated', device, self.current[0])

    def memory_reserved(self, device):
        return self._call('memory_reserved', device, self.current[1])

    def max_memory_allocated(self, device):
        return self._call('max_memory_allocated', device, self.peaks[0])

    def max_memory_reserved(self, device):
        return self._call('max_memory_reserved', device, self.peaks[1])


def monitor(tmp_path, cuda=None, **kwargs):
    return RequestMemoryMonitor(proc_root=proc_fixture(tmp_path),
                                torch=SimpleNamespace(cuda=cuda) if cuda else None,
                                **kwargs)


def test_analytical_units_scopes_and_snapshot_window(tmp_path):
    cuda = Cuda()
    ticks = iter(range(10, 100))
    mem = monitor(tmp_path, cuda, clock=lambda: next(ticks), device='cuda:3')
    before = mem.start()
    cuda.current = [120, 240]
    cuda.peaks = [300, 450]
    result = mem.finish()
    assert result['schema_version'] == 'kvolt-memory-v1'
    assert result['status'] == 'valid' and result['reason'] is None
    assert result['before'] == before
    assert before['system']['total_bytes'] == 1000 * 1024
    assert before['system']['available_bytes'] == 250 * 1024
    assert before['system']['used_bytes'] == 750 * 1024
    assert before['system']['usage_percent'] == 75.0
    assert before['system']['swap_used_bytes'] == 75 * 1024
    assert before['process']['rss_bytes'] == 100 * 1024
    assert before['process']['hwm_bytes'] == 150 * 1024
    assert before['process']['swap_bytes'] == 10 * 1024
    assert before['cuda']['max_allocated_bytes'] is None
    assert before['cuda']['peak_reason'] == 'before_request_peak_reset'
    assert result['after']['cuda']['max_allocated_bytes'] == 300
    assert result['after']['cuda']['max_reserved_bytes'] == 450
    assert before['read_start_ns'] < before['read_end_ns']
    assert before['read_end_ns'] < result['cuda_peak_reset']['read_start_ns']
    assert result['cuda_peak_reset']['read_end_ns'] < result['after']['read_start_ns']
    assert result['clock'] == 'injected-test-clock'
    assert result['units'] == {'memory': 'bytes', 'percentage': 'percent', 'time': 'ns'}
    assert 'lifetime' in result['scope']['process_peak']
    assert 'not' in result['scope']['overlap']
    assert 'warmup' in result['scope']['cuda_peak_window']
    assert all(device == 'cuda:3' for name, device in cuda.calls
               if name not in ('is_initialized', 'is_available'))
    json.dumps(result, allow_nan=False)


def test_peak_reset_isolated_per_request_and_finish_idempotent(tmp_path):
    cuda = Cuda()
    mem = monitor(tmp_path, cuda)
    mem.start()
    cuda.peaks = [800, 900]
    assert mem.finish()['after']['cuda']['max_allocated_bytes'] == 800
    calls = len(cuda.calls)
    assert mem.finish()['after']['cuda']['max_allocated_bytes'] == 800
    assert len(cuda.calls) == calls
    mem.start()
    assert mem.finish()['after']['cuda']['max_allocated_bytes'] == 100
    assert sum(name == 'reset_peak_memory_stats' for name, _ in cuda.calls) == 2


@pytest.mark.parametrize('replacement', ['MemAvailable: -1 kB', 'MemAvailable: 1 MB',
                                         'MemAvailable: 1.5 kB', 'MemAvailable: nope kB',
                                         'MemAvailable: 1 kB\nMemAvailable: 2 kB'])
def test_malformed_or_duplicate_metric_does_not_erase_other_metrics(tmp_path, replacement):
    mem = RequestMemoryMonitor(proc_root=proc_fixture(
        tmp_path, MEMINFO.replace('MemAvailable: 250 kB', replacement)))
    mem.start()
    result = mem.finish()
    system = result['after']['system']
    assert result['status'] == 'partial'
    assert system['available_bytes'] is None
    assert system['used_bytes'] is None and system['usage_percent'] is None
    assert system['total_bytes'] == 1024000
    assert system['swap_used_bytes'] == 75 * 1024
    assert system['errors']['available_bytes']


def test_missing_optional_fields_never_become_zero(tmp_path):
    mem = RequestMemoryMonitor(proc_root=proc_fixture(
        tmp_path, 'MemTotal: 1000 kB\nMemAvailable: 250 kB\n', 'VmRSS: 100 kB\n'))
    mem.start()
    result = mem.finish()
    assert result['after']['system']['usage_percent'] == 75
    assert result['after']['system']['swap_used_bytes'] is None
    assert result['after']['process']['hwm_bytes'] is None
    assert result['after']['process']['swap_bytes'] is None
    assert result['after']['process']['errors']['hwm_bytes'] == 'missing_field:VmHWM'


def test_inconsistent_and_zero_denominator_values(tmp_path):
    mem = RequestMemoryMonitor(proc_root=proc_fixture(
        tmp_path, 'MemTotal: 0 kB\nMemAvailable: 1 kB\nSwapTotal: 2 kB\nSwapFree: 3 kB\n',
        'VmRSS: 100 kB\nVmHWM: 99 kB\nVmSwap: 0 kB\n'))
    mem.start()
    result = mem.finish()
    assert result['after']['system']['available_bytes'] is None
    assert result['after']['system']['usage_percent'] is None
    assert result['after']['system']['swap_free_bytes'] is None
    assert result['after']['system']['swap_used_bytes'] is None
    assert result['after']['process']['hwm_bytes'] is None
    assert result['after']['process']['rss_bytes'] == 100 * 1024
    assert result['after']['process']['swap_bytes'] == 0
    json.dumps(result, allow_nan=False)


def test_proc_unavailable_and_no_torch(tmp_path):
    mem = RequestMemoryMonitor(proc_root=tmp_path / 'nonexistent')
    mem.start()
    result = mem.finish()
    assert result['status'] == 'unavailable'
    assert result['after']['system']['total_bytes'] is None
    assert result['after']['process']['rss_bytes'] is None
    assert result['after']['cuda']['reason'] == 'torch_not_provided'
    assert result['cuda_peak_reset']['status'] == 'unavailable'


def test_independent_read_failure_preserves_process_and_cuda(tmp_path):
    cuda = Cuda()
    mem = monitor(tmp_path, cuda)
    (tmp_path / 'meminfo').unlink()
    (tmp_path / 'meminfo').mkdir()  # Deterministic read error even as root.
    mem.start()
    result = mem.finish()
    assert result['status'] == 'partial'
    assert result['after']['system']['status'] == 'unavailable'
    assert result['after']['process']['rss_bytes'] == 102400
    assert result['after']['cuda']['allocated_bytes'] == 100


@pytest.mark.parametrize('initialized,available,reason', [
    (False, True, 'cuda_not_initialized'), (True, False, 'cuda_unavailable')])
def test_cuda_unavailable_never_initializes_context(tmp_path, initialized, available, reason):
    cuda = Cuda()
    cuda.initialized, cuda.available = initialized, available
    mem = monitor(tmp_path, cuda)
    mem.start()
    result = mem.finish()
    assert result['status'] == 'partial'
    assert result['after']['cuda']['reason'] == reason
    assert not any(name == 'reset_peak_memory_stats' for name, _ in cuda.calls)
    if not initialized:
        assert not any(name == 'is_available' for name, _ in cuda.calls)


def test_failed_reset_never_publishes_stale_peaks(tmp_path):
    cuda = Cuda()
    cuda.fail.add('reset_peak_memory_stats')
    mem = monitor(tmp_path, cuda)
    mem.start()
    result = mem.finish()
    after = result['after']['cuda']
    assert result['status'] == 'partial'
    assert result['cuda_peak_reset']['status'] == 'unavailable'
    assert after['allocated_bytes'] == 100
    assert after['max_allocated_bytes'] is None and after['max_reserved_bytes'] is None
    assert 'reset' in after['peak_reason']
    assert not any(name.startswith('max_memory_') for name, _ in cuda.calls)


def test_counter_failure_and_invalid_values_are_independent(tmp_path):
    cuda = Cuda()
    mem = monitor(tmp_path, cuda)
    mem.start()
    cuda.fail.add('memory_allocated')
    cuda.peaks = [-1, 500]
    result = mem.finish()
    after = result['after']['cuda']
    assert after['allocated_bytes'] is None and after['reserved_bytes'] == 200
    assert after['max_allocated_bytes'] is None and after['max_reserved_bytes'] == 500
    assert after['errors']['allocated_bytes']
    assert after['errors']['max_allocated_bytes'] == 'invalid_nonnegative_integer'


@pytest.mark.parametrize('value', [True, 3.5, float('nan'), -1])
def test_invalid_cuda_integer_rejected(tmp_path, value):
    cuda = Cuda()
    mem = monitor(tmp_path, cuda)
    mem.start()
    cuda.current[0] = value
    result = mem.finish()
    assert result['after']['cuda']['allocated_bytes'] is None
    json.dumps(result, allow_nan=False)


def test_no_memory_and_finish_without_start(tmp_path):
    result = no_memory()
    assert result['schema_version'] == 'kvolt-memory-v1'
    assert result['status'] == 'not_run' and result['reason'] == 'not_run'
    assert result['before'] is None and result['after'] is None
    assert RequestMemoryMonitor(proc_root=tmp_path).finish()['reason'] == 'not_started'
    assert no_memory('monitor_failed')['reason'] == 'monitor_failed'
    json.dumps(result, allow_nan=False)


def test_no_torch_import_or_import_time_proc_cuda_side_effects():
    path = Path(__file__).parents[1] / 'src/kvolt_runtime/telemetry/memory.py'
    script = '''
import builtins, importlib.util, pathlib, sys
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch imported')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
def no_read(*args, **kwargs):
    raise AssertionError('import-time filesystem read')
pathlib.Path.read_text = no_read
spec = importlib.util.spec_from_file_location('standalone_memory', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script, str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_clock_failure_retains_values_but_suppresses_unbounded_peaks(tmp_path):
    cuda = Cuda()

    def broken_clock():
        raise RuntimeError('clock failed')

    mem = monitor(tmp_path, cuda, clock=broken_clock)
    mem.start()
    result = mem.finish()
    assert result['status'] == 'partial'
    assert result['before']['read_start_ns'] is None
    assert result['after']['read_end_ns'] is None
    assert result['after']['system']['usage_percent'] == 75
    assert result['after']['cuda']['allocated_bytes'] == 100
    assert result['after']['cuda']['max_allocated_bytes'] is None
    assert result['cuda_peak_reset']['reason'] == 'invalid_reset_clock'


def test_cuda_probe_error_retains_host_values(tmp_path):
    class BrokenCuda:
        def is_initialized(self):
            raise RuntimeError('probe failed')

    mem = monitor(tmp_path, BrokenCuda())
    mem.start()
    result = mem.finish()
    assert result['status'] == 'partial'
    assert result['after']['cuda']['reason'] == 'cuda_probe_error:RuntimeError'
    assert result['after']['system']['used_bytes'] == 768000
    assert result['after']['process']['rss_bytes'] == 102400


def test_inconsistent_cuda_current_and_peak_metrics_rejected(tmp_path):
    cuda = Cuda()
    mem = monitor(tmp_path, cuda)
    mem.start()
    cuda.current = [100, 90]
    cuda.peaks = [99, 98]
    result = mem.finish()
    after = result['after']['cuda']
    assert after['allocated_bytes'] == 100
    assert after['reserved_bytes'] is None
    assert after['max_allocated_bytes'] is None
    assert after['max_reserved_bytes'] is None
    assert after['errors']['reserved_bytes'] == 'below_allocated_bytes'
    assert after['errors']['max_allocated_bytes'] == 'below_allocated_bytes'
    assert after['errors']['max_reserved_bytes'] == 'below_allocated_bytes'


def test_duplicate_process_field_and_zero_swap_total(tmp_path):
    mem = RequestMemoryMonitor(proc_root=proc_fixture(
        tmp_path, MEMINFO.replace('SwapTotal: 200 kB\nSwapFree: 125 kB',
                                  'SwapTotal: 0 kB\nSwapFree: 0 kB'),
        STATUS + 'VmRSS: 101 kB\n'))
    mem.start()
    after = mem.finish()['after']
    assert after['system']['swap_used_bytes'] == 0
    assert after['process']['rss_bytes'] is None
    assert after['process']['errors']['rss_bytes'] == 'duplicate_field:VmRSS'
    assert after['process']['hwm_bytes'] == 153600


def test_successful_reset_alone_is_not_a_memory_measurement(tmp_path):
    cuda = Cuda()
    cuda.fail.update(('memory_allocated', 'memory_reserved',
                      'max_memory_allocated', 'max_memory_reserved'))
    mem = RequestMemoryMonitor(proc_root=tmp_path / 'missing',
                                torch=SimpleNamespace(cuda=cuda))
    mem.start()
    result = mem.finish()
    assert result['cuda_peak_reset']['status'] == 'valid'
    assert result['before']['status'] == 'unavailable'
    assert result['after']['status'] == 'unavailable'
    assert result['status'] == 'unavailable'
    assert result['clock'] == 'time.monotonic_ns'
    assert no_memory()['clock'] == 'time.monotonic_ns'
