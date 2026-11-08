"""Persistent exact-request EMA observations over an immutable offline LUT.

Observed boundaries include controller/device/packing costs already measured by
the runner. This adaptive source does not inherit offline formal qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile

from .lut import _TabulatedLUT, _bracket
from .types import (LUTError, PERFORMANCE_LENGTHS, finite, frequency, group,
                    length as validate_length)
from ..errors import ContractError


SCHEMA = 'kvolt.online-lut.v1'
_METRICS = {'prefill': ('ttft_s', 'prefill_energy_j'),
            'decode': ('tpot_s', 'energy_j_per_token')}


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, OverflowError) as error:
        raise LUTError('online state requires finite JSON data') from error


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LUTError('duplicate online state JSON key')
        result[key] = value
    return result


def _number(value, *, positive=False):
    try:
        return finite(value, 'online observation', positive=positive)
    except ContractError:
        return None


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _mapping(value):
    return value if type(value) is dict else {}


class _DurabilityError(LUTError):
    """Replacement committed, but directory durability could not be confirmed."""

    def __init__(self, message, disk_sha256):
        super().__init__(message)
        self.disk_sha256 = disk_sha256


class OnlineLUTRepository(_TabulatedLUT):
    """Keep independent metric EMAs at the exact observed L/f or L/G/f.

    Construction validates an existing state without writing. Updates use a
    nonblocking advisory lock and disk identity check; a stale writer must reopen
    the repository rather than discard another process's observations.
    """

    def __init__(self, base, *, state_path, context=None, alpha=0.1):
        if not isinstance(base, _TabulatedLUT) or isinstance(base, OnlineLUTRepository):
            raise LUTError('online base must be a non-adaptive tabulated LUT')
        alpha = _number(alpha, positive=True)
        if alpha is None or alpha > 1:
            raise LUTError('online alpha must be finite and 0 < alpha <= 1')
        if context is not None and type(context) is not dict:
            raise LUTError('online context must be a JSON object')
        self._alpha = alpha
        # Preserve lexical path until lstat checks: resolving would hide symlinks.
        self.state_path = Path(os.path.abspath(os.fspath(state_path)))
        self._base = base
        for name in ('_prefill', '_decode', '_quality', '_prefill_target', '_decode_target'):
            setattr(self, name, getattr(base, name))
        self._prefill_time_monotone = False
        self.provenance = replace(base.provenance, kind='adaptive', notes=(*base.provenance.notes,
            'Request-measured online EMA overlay; no inherited measured/formal qualification.'))
        tables = {name: [[*key, *value] if isinstance(key, tuple) else [key, value]
                         for key, value in sorted(getattr(base, name).items())]
                  for name in ('_prefill', '_decode')}
        for name in ('_quality', '_prefill_target', '_decode_target'):
            tables[name] = [[*(key if isinstance(key, tuple) else (key,)), value]
                            for key, value in sorted(getattr(base, name).items())]
        self._binding = json.loads(_canonical({'base': {'provenance': asdict(base.provenance),
                                               'tables_sha256': _sha(tables)},
                                               'alpha': alpha, 'context': context or {}}))
        self._binding_sha256 = _sha(self._binding)
        self._state = {'schema': SCHEMA, 'version': 1, 'binding': self._binding,
                       'binding_sha256': self._binding_sha256, 'alpha': alpha,
                       'revision': 0, 'observation_ids': [],
                       'cells': {'prefill': {}, 'decode': {}}}
        self._state['state_sha256'] = _sha(self._state)
        raw = self._read_disk()
        if raw is not None:
            self._state = self._validate_state(raw)
        self._disk_sha256 = hashlib.sha256(raw).hexdigest() if raw is not None else None

    @property
    def base(self):
        """The immutable baseline used to initialize unseen observation metrics."""
        return self._base

    @property
    def alpha(self):
        """The bound learning rate cannot change while a state is open."""
        return self._alpha

    def _prefill_value(self, length, frequency_mhz, metric, bracket):
        cell = self._state['cells']['prefill'].get(f'{length}:{frequency_mhz}', {})
        entry = cell.get(_METRICS['prefill'][metric])
        return entry['ema'] if entry else super()._prefill_value(length, frequency_mhz, metric, bracket)

    def _decode_value(self, length, group, frequency_mhz, metric, bracket):
        cell = self._state['cells']['decode'].get(f'{length}:{group}:{frequency_mhz}', {})
        entry = cell.get(_METRICS['decode'][metric])
        return entry['ema'] if entry else super()._decode_value(length, group, frequency_mhz, metric, bracket)

    def snapshot(self):
        """Return stable, detached evidence without serializing all learned cells."""
        return {'schema': SCHEMA, 'version': 1, 'state_path': str(self.state_path),
                'revision': self._state['revision'], 'state_sha256': self._state['state_sha256'],
                'binding_sha256': self._binding_sha256, 'alpha': self.alpha,
                'observation_count': len(self._state['observation_ids']),
                'prefill_cells': len(self._state['cells']['prefill']),
                'decode_cells': len(self._state['cells']['decode']),
                'kind': 'adaptive', 'formal_qualification_inherited': False,
                'length_scope': 'exact observed L only',
                'decode_output_length_scope': 'mixed actual N; no output-length estimator/bucket'}

    def export_state(self):
        """Return the full detached canonical state, suitable for exact replay."""
        return deepcopy(self._state)

    def _safe_path(self, path):
        for component in (path, *path.parents):
            try:
                mode = component.lstat().st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise LUTError('online state path must not contain a symlink')

    def _read_disk(self):
        self._safe_path(self.state_path)
        try:
            descriptor = os.open(self.state_path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise LUTError(f'cannot read online state: {error}') from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise LUTError('online state must be a regular file')
            with os.fdopen(descriptor, 'rb') as stream:
                descriptor = None
                return stream.read()
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _validate_state(self, raw):
        try:
            state = json.loads(raw, object_pairs_hook=_json_object,
                               parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            expected = {'schema', 'version', 'binding', 'binding_sha256', 'alpha', 'revision',
                        'observation_ids', 'cells', 'state_sha256'}
            if type(state) is not dict or set(state) != expected:
                raise ValueError('unexpected state keys')
            if state['schema'] != SCHEMA or type(state['version']) is not int or state['version'] != 1:
                raise ValueError('unsupported schema/version')
            payload = {k: v for k, v in state.items() if k != 'state_sha256'}
            if state['state_sha256'] != _sha(payload):
                raise ValueError('state checksum mismatch')
            if (state['binding_sha256'] != self._binding_sha256 or
                    _sha(state['binding']) != self._binding_sha256 or
                    type(state['alpha']) not in (int, float) or state['alpha'] != self.alpha):
                raise ValueError('state binding mismatch')
            ids = state['observation_ids']
            if (type(ids) is not list or any(type(i) is not str or not i for i in ids)
                    or len(set(ids)) != len(ids) or not _integer(state['revision'])
                    or state['revision'] != len(ids)):
                raise ValueError('invalid revision/observation IDs')
            if type(state['cells']) is not dict or set(state['cells']) != set(_METRICS):
                raise ValueError('invalid cells')
            for phase, metrics in _METRICS.items():
                table = state['cells'][phase]
                if type(table) is not dict:
                    raise ValueError('cell table must be an object')
                for key, cell in table.items():
                    parts = key.split(':')
                    coordinates = tuple(int(part) for part in parts)
                    if ':'.join(map(str, coordinates)) != key or len(parts) != (2 if phase == 'prefill' else 3):
                        raise ValueError('noncanonical cell key')
                    validate_length(coordinates[0]); frequency(coordinates[-1])
                    if phase == 'decode': group(coordinates[1])
                    if type(cell) is not dict or not cell or not set(cell) <= set(metrics):
                        raise ValueError('unknown or empty metric cell')
                    for metric, entry in cell.items():
                        fields = {'ema', 'count'} | ({'last_n', 'min_n', 'max_n'} if phase == 'decode' else set())
                        if type(entry) is not dict or set(entry) != fields:
                            raise ValueError('unexpected metric keys')
                        if (_number(entry['ema'], positive=metric == metrics[0]) is None
                                or not _integer(entry['count'], 1) or entry['count'] > state['revision']):
                            raise ValueError('invalid metric value/count')
                        if phase == 'decode' and (not all(_integer(entry[k], 2) and entry[k] <= 512
                                for k in ('last_n', 'min_n', 'max_n'))
                                or not entry['min_n'] <= entry['last_n'] <= entry['max_n']):
                            raise ValueError('invalid Decode output counts')
            if bool(ids) != any(state['cells'].values()):
                raise ValueError('empty observations/cells disagree')
            return state
        except (ValueError, TypeError, KeyError, OverflowError, ContractError, UnicodeError) as error:
            raise LUTError(f'invalid online state: {error}') from error

    @contextmanager
    def _locked(self):
        self._safe_path(self.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_path.with_name(self.state_path.name + '.lock')
        self._safe_path(lock_path)
        descriptor = None
        try:
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise LUTError('online state lock must be a regular file')
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        except OSError as error:
            raise LUTError(f'online state lock/save failed: {error}') from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _save(self, state):
        raw = _canonical(state)
        temporary = None
        committed = False
        try:
            descriptor, temporary = tempfile.mkstemp(prefix='.'+self.state_path.name+'.', dir=self.state_path.parent)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            self._safe_path(self.state_path)
            os.replace(temporary, self.state_path)
            temporary = None
            committed = True
            directory = os.open(self.state_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            if committed:
                raise _DurabilityError(
                    f'online state save committed but directory durability uncertain: {error}',
                    hashlib.sha256(raw).hexdigest()) from error
            raise LUTError(f'online state save failed: {error}') from error
        finally:
            if temporary is not None:
                os.unlink(temporary)
        return hashlib.sha256(raw).hexdigest()

    def _observations(self, result):
        reasons, updates = [], []
        result = _mapping(result)
        generation, timing = _mapping(result.get('generation')), _mapping(result.get('timing'))
        if generation.get('status') != 'completed' or timing.get('valid') is not True:
            return [], ['incomplete_generation_or_invalid_timing']
        ttft = _number(timing.get('ttft_s'), positive=True)
        n = generation.get('n_output_tokens')
        if ttft is None or not _integer(n, 1) or n > 512:
            return [], ['invalid_ttft_or_output_count']
        for field in ('token_ids', 'token_ready_ns'):
            if field in generation and (type(generation[field]) is not list or len(generation[field]) != n):
                return [], ['generation_output_count_mismatch']
        for field, expected in (('n_output_tokens', n), ('n_decode_tokens', n-1)):
            if field in timing and (type(timing[field]) is not int or timing[field] != expected):
                return [], ['timing_output_count_mismatch']
        if any(k in timing for k in ('t0_ns', 't1_ns', 'tN_ns')):
            start, first, last = (timing.get(k) for k in ('t0_ns', 't1_ns', 'tN_ns'))
            if (not all(_integer(v) for v in (start, first, last)) or not start < first <= last
                    or (n == 1) != (first == last)
                    or not math.isclose(ttft, (first-start)/1e9, rel_tol=1e-12, abs_tol=1e-12)):
                return [], ['timing_boundary_mismatch']
        request = _mapping(result.get('request'))
        decisions, device = _mapping(result.get('decisions')), _mapping(result.get('device'))
        selected = _mapping(_mapping(decisions.get('prefill')).get('selected'))
        try:
            length = validate_length(request.get('length'))
            frequency_mhz = frequency(selected.get('frequency_mhz'))
            if type(selected.get('length')) is not int or selected['length'] != length:
                raise ContractError('selected length differs')
        except ContractError:
            return [], ['invalid_prefill_selection']
        if not self._verified(device.get('prefill'), frequency_mhz):
            return [], ['unverified_prefill_frequency']
        updates.append(('prefill', (length, frequency_mhz), 'ttft_s', ttft, None))
        energy = _mapping(result.get('energy'))
        prefill_energy = _number(energy.get('prefill_energy_j')) if energy.get('valid') is True else None
        if prefill_energy is not None:
            updates.append(('prefill', (length, frequency_mhz), 'prefill_energy_j', prefill_energy, None))
        else:
            reasons.append('prefill_energy_unavailable_or_invalid')
        if n == 1:
            return updates, [*reasons, 'no_decode_tokens']
        decode = _mapping(decisions.get('decode')); selected = _mapping(decode.get('selected'))
        try:
            decode_frequency, decode_group = frequency(selected.get('frequency_mhz')), group(selected.get('group'))
            if decode.get('status') == 'skipped' or type(selected.get('length')) is not int or selected['length'] != length:
                raise ContractError('invalid Decode selection')
        except ContractError:
            return updates, [*reasons, 'invalid_decode_selection']
        if not self._verified(device.get('decode'), decode_frequency):
            return updates, [*reasons, 'unverified_decode_frequency']
        tpot = _number(timing.get('mean_tpot_s'), positive=True)
        expected_tpots = []
        if 'decode_duration_s' in timing:
            duration = _number(timing['decode_duration_s'], positive=True)
            if duration is None:
                tpot = None
                reasons.append('invalid_decode_duration')
            else:
                expected_tpots.append(duration/(n-1))
        if 'tN_ns' in timing:
            expected_tpots.append((timing['tN_ns']-timing['t1_ns'])/(n-1)/1e9)
        if tpot is not None and any(not math.isclose(tpot, expected, rel_tol=1e-12, abs_tol=1e-12)
                                    for expected in expected_tpots):
            tpot = None
            reasons.append('decode_normalization_mismatch')
        coordinates = (length, decode_group, decode_frequency)
        if tpot is not None:
            updates.append(('decode', coordinates, 'tpot_s', tpot, n))
        else:
            reasons.append('invalid_decode_timing')
        decode_energy = _number(energy.get('decode_j_per_token')) if energy.get('valid') is True else None
        if decode_energy is not None and 'decode_energy_j' in energy:
            total = _number(energy['decode_energy_j'])
            if total is None or not math.isclose(decode_energy, total/(n-1), rel_tol=1e-12, abs_tol=1e-12):
                decode_energy = None
        if decode_energy is not None:
            updates.append(('decode', coordinates, 'energy_j_per_token', decode_energy, n))
        else:
            reasons.append('decode_energy_unavailable_or_invalid')
        return updates, reasons

    @staticmethod
    def _verified(receipt, frequency_mhz):
        receipt = _mapping(receipt)
        if receipt.get('verified') is not True:
            return False
        # Production device receipts expose target_hz; fixtures may only carry
        # the runner's mandatory verified flag.
        for field, expected in (('target_hz', frequency_mhz*1_000_000),
                                ('frequency_mhz', frequency_mhz)):
            if field in receipt and (type(receipt[field]) is not int or receipt[field] != expected):
                return False
        return True

    def observe(self, result, *, observation_id):
        """Commit one completed measurement, independently gating each metric."""
        if type(observation_id) is not str or not observation_id:
            raise LUTError('observation_id must be a nonempty string')
        before = self._state['revision']
        receipt = {'observation_id': observation_id, 'before_revision': before,
                   'after_revision': before, 'status': 'skipped', 'updated': [], 'reasons': []}
        if observation_id in self._state['observation_ids']:
            receipt['status'] = 'duplicate'
            return receipt
        updates, reasons = self._observations(result)
        receipt['reasons'] = reasons
        if not updates:
            return receipt
        with self._locked():
            raw = self._read_disk()
            disk_sha = hashlib.sha256(raw).hexdigest() if raw is not None else None
            if disk_sha != self._disk_sha256:
                raise LUTError('stale online state writer; reopen the repository')
            state = deepcopy(self._state)
            for phase, coordinates, metric, actual, n in updates:
                key = ':'.join(map(str, coordinates))
                cell = state['cells'][phase].setdefault(key, {})
                previous = cell.get(metric)
                index = _METRICS[phase].index(metric)
                bracket = _bracket(coordinates[0], PERFORMANCE_LENGTHS)
                baseline = (self._base._prefill_value(*coordinates, index, bracket) if phase == 'prefill'
                            else self._base._decode_value(*coordinates, index, bracket))
                old = previous['ema'] if previous else baseline
                entry = {'ema': (1-self.alpha)*old+self.alpha*actual,
                         'count': previous['count']+1 if previous else 1}
                if n is not None:
                    entry.update(last_n=n, min_n=min(previous['min_n'], n) if previous else n,
                                 max_n=max(previous['max_n'], n) if previous else n)
                cell[metric] = entry
                receipt['updated'].append(f'{phase}:{key}:{metric}')
            state['revision'] += 1
            state['observation_ids'].append(observation_id)
            state.pop('state_sha256')
            state['state_sha256'] = _sha(state)
            try:
                disk_sha = self._save(state)
            except _DurabilityError as error:
                # Rename already published these bytes. Keep memory aligned with
                # the actual committed revision and stop the caller on uncertain
                # crash durability instead of reporting an apparent rollback.
                self._state, self._disk_sha256 = state, error.disk_sha256
                error.snapshot = self.snapshot()
                raise
            # No memory mutation before durable save: failures retain all prior
            # predictions, revision, count and duplicate-observation evidence.
            self._state, self._disk_sha256 = state, disk_sha
        receipt.update(status='updated', after_revision=state['revision'])
        return receipt
