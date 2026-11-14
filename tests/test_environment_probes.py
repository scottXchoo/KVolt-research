"""Synthetic Linux sensors exercise observation without touching real hardware."""
from pathlib import Path

import pytest

from experiment_control.probes import LinuxProbe


def put(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value))
    return path


@pytest.fixture
def linux(tmp_path):
    sys_root, proc_root = tmp_path / 'sys', tmp_path / 'proc'
    sys_root.mkdir()
    proc_root.mkdir()
    put(proc_root, 'stat', 'cpu 100 0 50 850 0 0 0 0 0 0\n')
    put(proc_root, 'meminfo', 'MemTotal: 1000 kB\nMemAvailable: 400 kB\nSwapTotal: 200 kB\nSwapFree: 150 kB\n')
    put(proc_root, 'vmstat', 'pswpin 7\npswpout 9\n')
    put(proc_root, 'loadavg', '0.25 0.50 0.75 1/11 123\n')
    put(proc_root, 'pressure/cpu', 'some avg10=0.25 avg60=0.10 avg300=0.00 total=12\n')
    put(proc_root, 'pressure/memory', 'some avg10=1.50 avg60=0.00 avg300=0.00 total=1\nfull avg10=0.0 avg60=0.0 avg300=0.0 total=0\n')
    (proc_root / '123').mkdir()
    (proc_root / '456').mkdir()
    put(sys_root, 'devices/system/cpu/online', '0-3\n')
    put(sys_root, 'devices/system/cpu/cpufreq/policy0/scaling_cur_freq', '2201600')
    put(sys_root, 'devices/system/cpu/cpufreq/policy0/scaling_min_freq', '729600')
    put(sys_root, 'devices/system/cpu/cpufreq/policy0/scaling_max_freq', '2201600')
    put(sys_root, 'devices/system/cpu/cpufreq/policy0/scaling_governor', 'schedutil')
    return sys_root, proc_root


def probe(linux, **kwargs):
    return LinuxProbe(sys_root=linux[0], proc_root=linux[1], **kwargs)


def test_linux_units_and_sources(linux):
    sys_root, _ = linux
    thermal = put(sys_root, 'class/thermal/thermal_zone0/temp', '47500')
    put(sys_root, 'class/thermal/thermal_zone0/type', 'GPU-therm')
    gpu = sys_root / 'gpu'
    for key, value in [('cur_freq', 816000000), ('min_freq', 408000000), ('max_freq', 1122000000), ('governor', 'userspace')]:
        put(gpu, key, value)
    emc = put(sys_root, 'emc_rate', 3199000000)
    result = probe(linux, gpu_path=gpu, emc_path=emc, clock_ns=lambda: 123).sample()
    assert result.values['sample.monotonic_ns'] == 123
    assert result.values['temperature.GPU-therm.c'] == 47.5
    assert result.values['source.temperature.GPU-therm.c'] == str(thermal)
    assert result.values['clock.cpu.policy0.current_hz'] == 2201600000
    assert result.values['clock.cpu.policy0.governor'] == 'schedutil'
    assert result.values['clock.gpu.current_hz'] == 816000000
    assert result.values['clock.emc.hz'] == 3199000000
    assert result.values['memory.available_bytes'] == 409600
    assert result.values['memory.total_bytes'] == 1024000
    assert result.values['memory.swap_used_bytes'] == 51200
    assert result.values['vm.pswpin'] == 7
    assert result.values['vm.pswpout'] == 9
    assert result.values['process.count'] == 2
    assert result.values['cpu.online'] == '0-3'
    assert result.values['load.1m'] == 0.25
    assert result.values['pressure.cpu.some.avg10'] == 0.25
    assert result.values['pressure.memory.some.avg10'] == 1.5


def test_cpu_first_sample_delta_guest_excluded_and_reset(linux):
    item = probe(linux)
    first = item.sample()
    assert first.values['cpu.utilization_pct'] is None
    assert first.errors['cpu.utilization_pct'] == 'first_sample'
    put(linux[1], 'stat', 'cpu 120 0 60 920 0 0 0 0 10 0\n')
    assert item.sample().values['cpu.utilization_pct'] == 30
    put(linux[1], 'stat', 'cpu 1 0 0 9 0 0 0 0 0 0\n')
    reset = item.sample()
    assert reset.values['cpu.utilization_pct'] is None
    assert reset.errors['cpu.utilization_pct'] == 'counter_reset'
    put(linux[1], 'stat', 'cpu 2 0 0 18 0 0 0 0 0 0\n')
    assert item.sample().values['cpu.utilization_pct'] == 10


def test_duplicate_thermal_and_fan_channels_preserved(linux):
    for index, value in [(0, 40000), (1, 42000)]:
        put(linux[0], f'class/thermal/thermal_zone{index}/type', 'CPU')
        put(linux[0], f'class/thermal/thermal_zone{index}/temp', value)
    put(linux[0], 'class/hwmon/hwmon2/name', 'fan-controller')
    put(linux[0], 'class/hwmon/hwmon2/fan1_input', 1200)
    put(linux[0], 'class/hwmon/hwmon2/fan2_input', 1500)
    put(linux[0], 'class/hwmon/hwmon2/pwm1', 128)
    put(linux[0], 'class/thermal/cooling_device0/type', 'thermal-cpufreq')
    put(linux[0], 'class/thermal/cooling_device0/cur_state', 2)
    result = probe(linux).sample()
    assert result.values['temperature.CPU.0.c'] == 40
    assert result.values['temperature.CPU.1.c'] == 42
    assert result.values['fan.hwmon2.fan1.rpm'] == 1200
    assert result.values['fan.hwmon2.fan2.rpm'] == 1500
    assert result.values['fan.hwmon2.pwm1.pwm'] == 128
    assert result.values['cooling.cooling_device0.state'] == 2
    assert result.values['cooling.cooling_device0.type'] == 'thermal-cpufreq'
    assert result.values['throttle.status'] == 'unavailable'
    assert result.errors['throttle'] == 'source_not_configured'


def test_missing_roots_are_explicit(tmp_path):
    result = LinuxProbe(sys_root=tmp_path / 'no-sys', proc_root=tmp_path / 'no-proc').sample()
    for key in ['cpu.online', 'memory.total_bytes', 'load.1m', 'process.count', 'cpu.utilization_pct', 'clock.gpu.current_hz', 'clock.emc.hz']:
        assert result.values[key] is None
        assert key in result.errors
    assert result.values['temperature.status'] is None
    assert 'temperature.status' in result.errors


def test_invalid_values_never_become_zero_or_nonfinite(linux):
    put(linux[0], 'class/thermal/thermal_zone0/type', 'GPU')
    put(linux[0], 'class/thermal/thermal_zone0/temp', 'NaN')
    put(linux[1], 'loadavg', 'inf 0 0 1/1 1')
    put(linux[1], 'meminfo', 'MemTotal: -1 kB\nMemAvailable: 3 MB\nSwapTotal: 1 kB\nSwapFree: 2 kB\n')
    result = probe(linux).sample()
    for key in ['temperature.GPU.c', 'load.1m', 'memory.total_bytes', 'memory.available_bytes', 'memory.swap_used_bytes']:
        assert result.values[key] is None
        assert key in result.errors


def test_explicit_custom_sensors_and_unknown_gpu_load(linux):
    gpu = linux[0] / 'gpu'
    load = put(gpu, 'load', 750)
    counter = put(linux[0], 'oc_counter', 3)
    voltage = put(linux[0], 'supply_mv', 19000)
    result = probe(linux, gpu_path=gpu, extra_metrics={
        'gpu.utilization_pct': {'path': load, 'scale': 0.1, 'kind': 'float'},
        'throttle.oc.count': {'path': counter, 'kind': 'int'},
        'supply.voltage_v': {'path': voltage, 'scale': 0.001},
    }).sample()
    assert result.values['gpu.utilization_pct'] == 75
    assert result.values['throttle.oc.count'] == 3
    assert result.values['supply.voltage_v'] == 19
    assert result.values['throttle.status'] == 'configured'
    assert 'throttle' not in result.errors
    unknown = probe(linux, gpu_path=gpu).sample()
    assert unknown.values['gpu.utilization_pct'] is None
    assert unknown.errors['gpu.utilization_pct'] == 'source_not_configured'


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf'), True, '1'])
def test_custom_scale_must_be_positive_finite(linux, scale):
    with pytest.raises(ValueError):
        probe(linux, extra_metrics={'x': {'path': '/no-sensor', 'scale': scale}})


def test_custom_invalid_and_unreadable_sources(linux, monkeypatch):
    invalid = put(linux[0], 'invalid', 'inf')
    denied = put(linux[0], 'denied', 4)
    original = Path.read_text
    def read_text(path, *args, **kwargs):
        if path == denied:
            raise PermissionError('synthetic denial')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_text)
    result = probe(linux, extra_metrics={
        'test.invalid': {'path': invalid},
        'test.denied': {'path': denied},
    }).sample()
    assert result.values['test.invalid'] is None
    assert 'invalid' in result.errors['test.invalid']
    assert result.values['test.denied'] is None
    assert 'PermissionError' in result.errors['test.denied']


def test_custom_key_collision_rejected(linux):
    with pytest.raises(ValueError):
        probe(linux, extra_metrics={'memory.total_bytes': {'path': '/x'}}).sample()


def test_cpu_malformed_sample_clears_baseline(linux):
    item = probe(linux)
    item.sample()
    put(linux[1], 'stat', 'cpu nope 0 0 0')
    assert item.sample().values['cpu.utilization_pct'] is None
    put(linux[1], 'stat', 'cpu 200 0 0 900 0 0 0 0')
    assert item.sample().errors['cpu.utilization_pct'] == 'first_sample'


def test_cpu_actual_frequency_preferred_and_source_preserved(linux):
    put(linux[0], 'devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq', '2000000')
    result = probe(linux).sample()
    assert result.values['clock.cpu.policy0.current_hz'] == 2000000000
    assert result.values['clock.cpu.policy0.current_source'] == 'cpuinfo_cur_freq'
    assert result.values['source.clock.cpu.policy0.current_hz'].endswith('/cpuinfo_cur_freq')


def test_cpu_zero_delta_and_per_counter_reset(linux):
    item = probe(linux)
    item.sample()
    assert item.sample().errors['cpu.utilization_pct'] == 'no_elapsed_cpu_ticks'
    # Total rises, but one individual counter resets: do not publish a percent.
    put(linux[1], 'stat', 'cpu 99 0 60 900 0 0 0 0')
    assert item.sample().errors['cpu.utilization_pct'] == 'counter_reset'


def test_thermal_names_cannot_collide_with_generated_alias(linux):
    for index, name in enumerate(('CPU', 'CPU', 'CPU.0')):
        put(linux[0], f'class/thermal/thermal_zone{index}/type', name)
        put(linux[0], f'class/thermal/thermal_zone{index}/temp', 10000 * (index + 1))
    result = probe(linux).sample()
    assert result.values['temperature.CPU.0.c'] == 10
    assert result.values['temperature.CPU.1.c'] == 20
    assert result.values['temperature.CPU%2E0.c'] == 30


def test_emc_load_requires_explicit_units(linux):
    assert probe(linux).sample().values['emc.utilization_pct'] is None
    source = put(linux[0], 'emc_load', 25)
    result = probe(linux, extra_metrics={'emc.utilization_pct': {'path': source}}).sample()
    assert result.values['emc.utilization_pct'] == 25


def test_clock_failure_preserves_other_sensors(linux):
    result = probe(linux, clock_ns=lambda: float('nan')).sample()
    assert result.values['sample.monotonic_ns'] is None
    assert 'sample.monotonic_ns' in result.errors
    assert result.values['memory.total_bytes'] == 1024000
