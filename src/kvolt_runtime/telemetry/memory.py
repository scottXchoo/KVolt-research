"""Best-effort request-boundary memory diagnostics, independent of timing validity.

The caller must run ``start`` before t0 and ``finish`` after the final token's
ready timestamp. No polling, CUDA synchronization, context initialization, or
cache eviction is performed. An already loaded torch object is optional.
"""
from pathlib import Path
import re
import time


_SYSTEM_FIELDS = {
    'total_bytes': 'MemTotal', 'available_bytes': 'MemAvailable',
    'swap_total_bytes': 'SwapTotal', 'swap_free_bytes': 'SwapFree',
}
_PROCESS_FIELDS = {'rss_bytes': 'VmRSS', 'hwm_bytes': 'VmHWM', 'swap_bytes': 'VmSwap'}
_CUDA_FIELDS = {
    'allocated_bytes': 'memory_allocated', 'reserved_bytes': 'memory_reserved',
    'max_allocated_bytes': 'max_memory_allocated',
    'max_reserved_bytes': 'max_memory_reserved',
}


def _error(exc):
    # Error types are stable and avoid persisting arbitrary exception payloads.
    return type(exc).__name__


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _state(values, errors):
    present = sum(value is not None for value in values)
    return 'unavailable' if not present else ('partial' if errors else 'valid')


def _section(values, errors):
    return {**values, 'status': _state(values.values(), errors),
            'reason': ';'.join(f'{key}:{value}' for key, value in errors.items()) or None,
            'errors': errors}


def _combine(sections):
    statuses = [section['status'] for section in sections]
    if all(status == 'valid' for status in statuses):
        return 'valid'
    return 'unavailable' if all(status == 'unavailable' for status in statuses) else 'partial'


def _metadata(clock='time.monotonic_ns'):
    return {
        'schema_version': 'kvolt-memory-v1',
        'clock': clock,
        'units': {'memory': 'bytes', 'percentage': 'percent', 'time': 'ns'},
        'scope': {
            'snapshot_window': 'before t0 and after tN; boundary snapshots, not continuous peaks',
            'cuda_peak_window': 'successful reset before t0 through finish after tN; includes '
                                'resident model allocations, excludes earlier warmup peaks; '
                                'not an exact t0..tN window',
            'process_peak': 'VmHWM is the process lifetime RSS peak, not a request peak',
            'overlap': 'system RAM, process RSS and CUDA allocator memory overlap; do not add them',
            'cuda_allocator': 'PyTorch allocator bytes on the explicit device, not device-wide usage',
        },
    }


def no_memory(reason='not_run'):
    """Return an explicit JSON-native record when collection was not attempted."""
    return {**_metadata(), 'status': 'not_run' if reason == 'not_run' else 'unavailable',
            'reason': str(reason),
            'before': None, 'after': None,
            'cuda_peak_reset': {'status': 'unavailable', 'reason': str(reason),
                                'read_start_ns': None, 'read_end_ns': None}}


class RequestMemoryMonitor:
    """Collect independent host and optional CUDA memory values per request.

    A monitor may be reused sequentially; each ``start`` resets request peaks.
    ``finish`` is idempotent until the next start. CUDA diagnostics never import
    torch and require an available, already initialized CUDA backend.
    """

    def __init__(self, *, torch=None, clock=time.monotonic_ns, proc_root='/proc',
                 device='cuda:0'):
        self.torch = torch
        self.clock = clock
        self.proc_root = Path(proc_root)
        self.device = str(device)
        self.before = None
        self._record = None
        self._reset = {'status': 'unavailable', 'reason': 'not_started',
                       'read_start_ns': None, 'read_end_ns': None}

    def _now(self):
        try:
            value = self.clock()
            return value if _integer(value) else None
        except Exception:
            return None

    def _proc(self, relative, fields):
        values = dict.fromkeys(fields)
        errors = {}
        try:
            lines = (self.proc_root / relative).read_text(encoding='utf-8').splitlines()
        except Exception as exc:
            return _section(values, {key: f'read_error:{_error(exc)}' for key in fields})
        matches = {field: [] for field in fields.values()}
        for line in lines:
            name, separator, value = line.partition(':')
            if separator and name.strip() in matches:
                matches[name.strip()].append(value.strip())
        for key, field in fields.items():
            entries = matches[field]
            if not entries:
                errors[key] = f'missing_field:{field}'
            elif len(entries) != 1:
                errors[key] = f'duplicate_field:{field}'
            elif not re.fullmatch(r'[0-9]+\s+kB', entries[0]):
                errors[key] = f'invalid_kib_field:{field}'
            else:
                try:
                    values[key] = int(entries[0].split()[0]) * 1024
                except (ValueError, OverflowError):
                    errors[key] = f'invalid_kib_field:{field}'
        return _section(values, errors)

    def _system(self):
        raw = self._proc('meminfo', _SYSTEM_FIELDS)
        values = {key: raw[key] for key in _SYSTEM_FIELDS}
        errors = raw['errors'].copy()
        for total_key, free_key, used_key in (
                ('total_bytes', 'available_bytes', 'used_bytes'),
                ('swap_total_bytes', 'swap_free_bytes', 'swap_used_bytes')):
            total, free = values[total_key], values[free_key]
            values[used_key] = None
            if total is not None and free is not None:
                if free > total:
                    values[free_key] = None
                    errors[free_key] = f'exceeds_{total_key}'
                else:
                    values[used_key] = total - free
            if values[used_key] is None:
                errors[used_key] = 'source_metric_unavailable'
        values['usage_percent'] = None
        total, used = values['total_bytes'], values['used_bytes']
        if total is not None and total > 0 and used is not None:
            # Divide first to avoid overflowing a float for malformed huge ints.
            values['usage_percent'] = (used / total) * 100
        else:
            errors['usage_percent'] = 'nonpositive_total' if total == 0 else 'source_metric_unavailable'
        return _section(values, errors)

    def _process(self):
        raw = self._proc('self/status', _PROCESS_FIELDS)
        values = {key: raw[key] for key in _PROCESS_FIELDS}
        errors = raw['errors'].copy()
        if (values['rss_bytes'] is not None and values['hwm_bytes'] is not None
                and values['hwm_bytes'] < values['rss_bytes']):
            values['hwm_bytes'] = None
            errors['hwm_bytes'] = 'below_rss_bytes'
        return _section(values, errors)

    def _cuda_backend(self):
        if self.torch is None:
            return None, 'torch_not_provided'
        try:
            cuda = self.torch.cuda
            # Check initialization first: even availability checks must not be
            # needed on a CPU-only path that has never initialized CUDA.
            if not cuda.is_initialized():
                return None, 'cuda_not_initialized'
            if not cuda.is_available():
                return None, 'cuda_unavailable'
            return cuda, None
        except Exception as exc:
            return None, f'cuda_probe_error:{_error(exc)}'

    def _cuda(self, *, after):
        cuda, unavailable = self._cuda_backend()
        values = dict.fromkeys(_CUDA_FIELDS)
        errors = {}
        peak_reason = None if after else 'before_request_peak_reset'
        requested = list(_CUDA_FIELDS) if after else list(_CUDA_FIELDS)[:2]
        if after and self._reset['status'] != 'valid':
            peak_reason = f"request_peak_reset_unavailable:{self._reset['reason']}"
        for key in requested:
            if unavailable:
                errors[key] = unavailable
            elif key.startswith('max_') and peak_reason:
                errors[key] = peak_reason
            else:
                try:
                    value = getattr(cuda, _CUDA_FIELDS[key])(device=self.device)
                    if _integer(value):
                        values[key] = value
                    else:
                        errors[key] = 'invalid_nonnegative_integer'
                except Exception as exc:
                    errors[key] = f'counter_error:{_error(exc)}'
        # These counters are not an atomic snapshot. Retain independent valid
        # values but reject an inconsistent metric instead of inventing a value.
        for lower, upper in (('allocated_bytes', 'reserved_bytes'),
                             ('allocated_bytes', 'max_allocated_bytes'),
                             ('allocated_bytes', 'max_reserved_bytes'),
                             ('reserved_bytes', 'max_reserved_bytes'),
                             ('max_allocated_bytes', 'max_reserved_bytes')):
            if values[lower] is not None and values[upper] is not None and values[upper] < values[lower]:
                values[upper] = None
                errors[upper] = f'below_{lower}'
        result = _section({key: values[key] for key in requested}, errors)
        result.update(values)
        result.update(device=self.device, peak_reason=peak_reason)
        if unavailable:
            result['reason'] = unavailable
        return result

    def _snapshot(self, *, after):
        start = self._now()
        # Each source is isolated so a missing /proc never suppresses CUDA.
        sections = {}
        for name, read in (('system', self._system), ('process', self._process),
                           ('cuda', lambda: self._cuda(after=after))):
            try:
                sections[name] = read()
            except Exception as exc:
                fields = ({**_SYSTEM_FIELDS, 'used_bytes': '', 'usage_percent': '', 'swap_used_bytes': ''}
                          if name == 'system' else _PROCESS_FIELDS if name == 'process' else _CUDA_FIELDS)
                sections[name] = _section(dict.fromkeys(fields),
                                           {key: f'collection_error:{_error(exc)}' for key in fields})
                if name == 'cuda':
                    sections[name].update(device=self.device, peak_reason='collection_error')
        end = self._now()
        clock_error = None
        if start is None or end is None or end < start:
            clock_error = 'invalid_snapshot_clock'
            if start is not None and end is not None and end < start:
                end = None
        status = _combine(sections.values())
        if clock_error and status == 'valid':
            status = 'partial'
        return {'read_start_ns': start, 'read_end_ns': end, 'status': status,
                'reason': clock_error, **sections}

    def start(self):
        """Capture before values then reset CUDA peaks; caller has not set t0."""
        self._record = None
        self.before = self._snapshot(after=False)
        start = self._now()
        cuda, reason = self._cuda_backend()
        if cuda is not None:
            try:
                cuda.reset_peak_memory_stats(device=self.device)
            except Exception as exc:
                reason = f'reset_error:{_error(exc)}'
        end = self._now()
        if reason is None and (start is None or end is None or end < start):
            reason = 'invalid_reset_clock'
        self._reset = {'status': 'valid' if reason is None else 'unavailable',
                       'reason': reason, 'read_start_ns': start, 'read_end_ns': end}
        return self.before

    def finish(self):
        """Capture after values outside t0..tN and return JSON-native diagnostics."""
        if self._record is not None:
            return self._record
        if self.before is None:
            return no_memory('not_started')
        after = self._snapshot(after=True)
        status = _combine((self.before, after))
        if status == 'valid' and self._reset['status'] != 'valid':
            status = 'partial'
        reasons = []
        for boundary, snapshot in (('before', self.before), ('after', after)):
            if snapshot['reason']:
                reasons.append(f"{boundary}:{snapshot['reason']}")
            for name in ('system', 'process', 'cuda'):
                if snapshot[name]['reason']:
                    reasons.append(f"{boundary}.{name}:{snapshot[name]['reason']}")
        if self._reset['reason']:
            reasons.append(f"cuda_peak_reset:{self._reset['reason']}")
        clock_name = 'time.monotonic_ns' if self.clock is time.monotonic_ns else 'injected-test-clock'
        self._record = {**_metadata(clock_name), 'status': status, 'reason': ';'.join(reasons) or None,
                        'before': self.before, 'after': after, 'cuda_peak_reset': self._reset.copy()}
        return self._record
