"""Read-only Linux observations with explicit units and missing-source errors.

No hardware model is assumed. CPU cpufreq uses Linux kHz, thermal zones use
millidegrees Celsius, and explicitly supplied devfreq paths use Hz. Device-
specific load, voltage, and throttling sensors require ``extra_metrics`` with
caller-confirmed units. Cooling state alone never establishes throttling.
"""
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import re
import time
from urllib.parse import quote

from .contracts import Reading, finite_number


def _segment(value):
    """Keep dotted metric segments unambiguous, including literal percent signs."""
    return quote(value, safe='-_').replace('.', '%2E')


class _Reader:
    def __init__(self):
        self.values = {}
        self.errors = {}

    def missing(self, metric, reason):
        self.values[metric] = None
        self.errors[metric] = reason

    def read(self, path, metric):
        self.values['source.' + metric] = str(path)
        try:
            value = path.read_text(encoding='utf-8').strip()
            if not value:
                raise ValueError('empty source')
            return value
        except (OSError, UnicodeError, ValueError) as exc:
            self.missing(metric, f'{type(exc).__name__}: {path}')
            return None

    def convert(self, value, metric, *, kind='float', scale=1, minimum=None):
        if value is None:
            return None
        try:
            if kind == 'text':
                converted = value
            else:
                converted = int(value) if kind == 'int' else float(value)
                # Do not cast integer counters to float and lose precision.
                converted = converted if scale == 1 else converted * scale
                finite_number(converted, metric, minimum=minimum)
            self.values[metric] = converted
            return converted
        except (ValueError, OverflowError, TypeError):
            self.missing(metric, 'invalid: expected finite ' + kind)
            return None

    def sensor(self, path, metric, *, kind='float', scale=1, minimum=None):
        return self.convert(self.read(path, metric), metric, kind=kind,
                            scale=scale, minimum=minimum)

    def directories(self, root, pattern, metric):
        self.values['source.' + metric] = str(root)
        try:
            paths = sorted(p for p in root.iterdir()
                           if re.fullmatch(pattern, p.name) and p.is_dir())
        except OSError as exc:
            self.missing(metric, f'{type(exc).__name__}: {root}')
            return []
        if not paths:
            self.missing(metric, 'source_not_found: ' + str(root))
        else:
            self.values[metric] = 'available'
        return paths


class LinuxProbe:
    """A detachable stdlib probe; only ``sample`` performs filesystem reads.

    ``gpu_path`` is an optional devfreq directory; ``emc_path`` is an optional
    Hz-valued file or devfreq directory. ``extra_metrics`` maps flat metric names
    to ``{path, scale=1, kind='float'}``; kinds are int, float, or text. Scale
    must be positive and finite. Custom metrics cannot replace built-in values,
    except the explicitly configurable GPU/EMC utilization placeholders.
    Every sensor includes its path in ``source.<metric>``. Missing/invalid
    readings remain None with an error. No sensor changes device state.
    """

    def __init__(self, name='linux', *, sys_root='/sys', proc_root='/proc',
                 gpu_path=None, emc_path=None, extra_metrics=None,
                 clock_ns=time.monotonic_ns):
        if not isinstance(name, str) or not name:
            raise ValueError('probe name must be nonempty')
        if not callable(clock_ns):
            raise ValueError('clock_ns must be callable')
        self.name = name
        self.sys_root, self.proc_root = Path(sys_root), Path(proc_root)
        self.gpu_path = Path(gpu_path) if gpu_path is not None else None
        self.emc_path = Path(emc_path) if emc_path is not None else None
        if extra_metrics is not None and not isinstance(extra_metrics, Mapping):
            raise ValueError('extra_metrics must be a mapping')
        self.extra_metrics = {}
        for metric, specification in (extra_metrics or {}).items():
            if (not isinstance(metric, str) or not metric or
                    metric.startswith('source.') or metric == 'throttle.status'):
                raise ValueError('invalid custom metric name')
            if not isinstance(specification, Mapping) or 'path' not in specification:
                raise ValueError('custom metric requires a path mapping')
            if set(specification) - {'path', 'scale', 'kind'}:
                raise ValueError('unknown custom metric option')
            scale = specification.get('scale', 1)
            finite_number(scale, 'scale')
            if scale <= 0:
                raise ValueError('scale must be positive')
            kind = specification.get('kind', 'float')
            if kind not in ('int', 'float', 'text'):
                raise ValueError('kind must be int, float, or text')
            if kind == 'text' and scale != 1:
                raise ValueError('text sensors cannot be scaled')
            self.extra_metrics[metric] = (Path(specification['path']), scale, kind)
        self.clock_ns = clock_ns
        self._cpu_previous = None

    def sample(self):
        reader = _Reader()
        try:
            stamp = self.clock_ns()
            if type(stamp) is not int or stamp < 0:
                raise ValueError('clock_ns must return nonnegative integer')
            finite_number(stamp, 'sample.monotonic_ns')
            reader.values['sample.monotonic_ns'] = stamp
        except Exception as exc:
            reader.missing('sample.monotonic_ns', type(exc).__name__ + ': clock_ns')
        self._thermal(reader)
        self._fans(reader)
        self._clocks(reader)
        self._cpu(reader)
        self._memory(reader)
        self._system(reader)
        for metric, (path, scale, kind) in self.extra_metrics.items():
            if metric in reader.values and metric not in ('gpu.utilization_pct', 'emc.utilization_pct'):
                raise ValueError('custom metric collides with built-in metric: ' + metric)
            reader.errors.pop(metric, None)
            reader.sensor(path, metric, kind=kind, scale=scale)
        if any(metric.startswith('throttle.') for metric in self.extra_metrics):
            reader.values['throttle.status'] = 'configured'
        else:
            reader.values['throttle.status'] = 'unavailable'
            reader.errors['throttle'] = 'source_not_configured'
        return Reading(reader.values, reader.errors)

    def _thermal(self, reader):
        root = self.sys_root / 'class/thermal'
        zones = reader.directories(root, r'thermal_zone\d+', 'temperature.status')
        typed = []
        for zone in zones:
            metric = 'temperature.' + zone.name + '.type'
            sensor_type = reader.sensor(zone / 'type', metric, kind='text')
            # An unreadable type remains visible under the actual zone ID.
            typed.append((zone, _segment(sensor_type) if sensor_type else zone.name))
        counts = Counter(alias for _, alias in typed)
        positions = Counter()
        for zone, alias in typed:
            if counts[alias] > 1:
                index = positions[alias]
                positions[alias] += 1
                alias = f'{alias}.{index}'
            reader.sensor(zone / 'temp', f'temperature.{alias}.c', scale=0.001)
        for cooling in reader.directories(root, r'cooling_device\d+', 'cooling.status'):
            prefix = 'cooling.' + cooling.name
            reader.sensor(cooling / 'type', prefix + '.type', kind='text')
            reader.sensor(cooling / 'cur_state', prefix + '.state', kind='int', minimum=0)

    def _fans(self, reader):
        root = self.sys_root / 'class/hwmon'
        devices = reader.directories(root, r'hwmon\d+', 'fan.status')
        found = False
        for device in devices:
            try:
                channels = sorted(device.iterdir())
            except OSError as exc:
                reader.missing('fan.' + device.name, f'{type(exc).__name__}: {device}')
                continue
            for channel in channels:
                match = re.fullmatch(r'(fan\d+)_input|(pwm\d+)', channel.name)
                if not match:
                    continue
                found = True
                kind, unit = (match[1], 'rpm') if match[1] else (match[2], 'pwm')
                reader.sensor(channel, f'fan.{device.name}.{kind}.{unit}', kind='int', minimum=0)
            reader.sensor(device / 'name', 'fan.' + device.name + '.name', kind='text')
        if devices and not found:
            reader.missing('fan.status', 'source_not_found: fan/pwm channels')

    def _clocks(self, reader):
        root = self.sys_root / 'devices/system/cpu'
        reader.sensor(root / 'online', 'cpu.online', kind='text')
        for policy in reader.directories(root / 'cpufreq', r'policy\d+', 'clock.cpu.status'):
            prefix = 'clock.cpu.' + policy.name
            # scaling_cur_freq can be the last driver-requested value. Retain
            # the source name so consumers cannot mistake it for confirmation.
            actual = policy / 'cpuinfo_cur_freq'
            current_source = 'cpuinfo_cur_freq' if actual.exists() else 'scaling_cur_freq'
            reader.values[prefix + '.current_source'] = current_source
            for source, metric in [(current_source, 'current_hz'),
                                   ('scaling_min_freq', 'min_hz'),
                                   ('scaling_max_freq', 'max_hz')]:
                reader.sensor(policy / source, prefix + '.' + metric, kind='int', scale=1000, minimum=0)
            reader.sensor(policy / 'scaling_governor', prefix + '.governor', kind='text')
        for source, metric in [('cur_freq', 'current_hz'), ('min_freq', 'min_hz'),
                               ('max_freq', 'max_hz'), ('governor', 'governor')]:
            key = 'clock.gpu.' + metric
            if self.gpu_path is None:
                reader.missing(key, 'source_not_configured')
            else:
                reader.sensor(self.gpu_path / source, key,
                              kind='text' if source == 'governor' else 'int',
                              minimum=None if source == 'governor' else 0)
        if self.emc_path is None:
            reader.missing('clock.emc.hz', 'source_not_configured')
        else:
            path = self.emc_path / 'cur_freq' if self.emc_path.is_dir() else self.emc_path
            reader.sensor(path, 'clock.emc.hz', kind='int', minimum=0)
        reader.missing('gpu.utilization_pct', 'source_not_configured')
        reader.missing('emc.utilization_pct', 'source_not_configured')

    def _cpu(self, reader):
        metric = 'cpu.utilization_pct'
        value = reader.read(self.proc_root / 'stat', metric)
        if value is None:
            self._cpu_previous = None
            return
        try:
            fields = next(line.split() for line in value.splitlines()
                          if line.split() and line.split()[0] == 'cpu')
            counters = tuple(int(part) for part in fields[1:9])
            if len(counters) < 4 or any(part < 0 for part in counters):
                raise ValueError('invalid CPU counters')
            previous, self._cpu_previous = self._cpu_previous, counters
            if previous is None:
                reader.missing(metric, 'first_sample')
                return
            if len(previous) != len(counters) or any(now < old for now, old in zip(counters, previous)):
                reader.missing(metric, 'counter_reset')
                return
            delta = tuple(now - old for now, old in zip(counters, previous))
            total = sum(delta)
            if total == 0:
                reader.missing(metric, 'no_elapsed_cpu_ticks')
                return
            idle = delta[3] + (delta[4] if len(delta) > 4 else 0)
            reader.values[metric] = (total - idle) / total * 100
        except (ValueError, StopIteration, OverflowError):
            self._cpu_previous = None
            reader.missing(metric, 'invalid: CPU counters')

    def _memory(self, reader):
        source = self.proc_root / 'meminfo'
        metrics = {'MemTotal': 'memory.total_bytes', 'MemAvailable': 'memory.available_bytes',
                   'SwapTotal': 'memory.swap_total_bytes', 'SwapFree': 'memory.swap_free_bytes'}
        text = reader.read(source, 'memory.status')
        entries = {}
        if text is not None:
            reader.values['memory.status'] = 'available'
            for line in text.splitlines():
                key, separator, raw = line.partition(':')
                if separator:
                    entries[key] = raw.split()
        for field, metric in metrics.items():
            reader.values['source.' + metric] = str(source)
            raw = entries.get(field)
            if raw is None:
                reader.missing(metric, reader.errors.get('memory.status', 'source_field_not_found: ' + field))
            elif len(raw) != 2 or raw[1] != 'kB':
                reader.missing(metric, 'invalid: expected kB')
            else:
                reader.convert(raw[0], metric, kind='int', scale=1024, minimum=0)
        total, free = (reader.values.get('memory.swap_' + name + '_bytes') for name in ('total', 'free'))
        metric = 'memory.swap_used_bytes'
        reader.values['source.' + metric] = str(source)
        if total is None or free is None or free > total:
            reader.missing(metric, 'invalid: swap total/free')
        else:
            reader.values[metric] = total - free
        text = reader.read(self.proc_root / 'vmstat', 'vm.status')
        if text is not None:
            reader.values['vm.status'] = 'available'
        fields = dict(line.split() for line in (text or '').splitlines() if len(line.split()) == 2)
        for field in ('pswpin', 'pswpout'):
            metric = 'vm.' + field
            reader.values['source.' + metric] = str(self.proc_root / 'vmstat')
            if field not in fields:
                reader.missing(metric, reader.errors.get('vm.status', 'source_field_not_found: ' + field))
            else:
                reader.convert(fields[field], metric, kind='int', minimum=0)

    def _system(self, reader):
        metric = 'load.1m'
        text = reader.read(self.proc_root / 'loadavg', metric)
        if text is not None:
            reader.convert(text.split()[0], metric, minimum=0)
        metric = 'process.count'
        reader.values['source.' + metric] = str(self.proc_root)
        try:
            reader.values[metric] = sum(1 for entry in self.proc_root.iterdir()
                                        if entry.name.isascii() and entry.name.isdigit() and entry.is_dir())
        except OSError as exc:
            reader.missing(metric, f'{type(exc).__name__}: {self.proc_root}')
        for resource in ('cpu', 'memory', 'io'):
            path = self.proc_root / 'pressure' / resource
            status = f'pressure.{resource}.status'
            text = reader.read(path, status)
            if text is not None:
                reader.values[status] = 'available'
            for category in ('some', 'full'):
                metric = f'pressure.{resource}.{category}.avg10'
                reader.values['source.' + metric] = str(path)
                if text is None:
                    reader.missing(metric, reader.errors[status])
                    continue
                try:
                    fields = next(line.split()[1:] for line in text.splitlines()
                                  if line.startswith(category + ' '))
                    values = dict(field.split('=', 1) for field in fields)
                    reader.convert(values['avg10'], metric, minimum=0)
                except (StopIteration, ValueError, KeyError):
                    reader.missing(metric, 'source_field_not_found: ' + category + '.avg10')
