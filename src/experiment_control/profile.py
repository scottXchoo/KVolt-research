"""Strict, explicit JSON profiles; loading a profile never changes hardware."""
import json
from pathlib import Path
import re

from .contracts import MetricRule
from .controls import (AffinityControl, ClockControl, EmcControl, FanControl,
                       PowerModeControl, SwappinessControl)
from .guard import Environment
from .lock import COMMON_LOCK, FileLock
from .probes import LinuxProbe


def _fields(value, allowed, label):
    if type(value) is not dict or set(value) - set(allowed):
        raise ValueError(f'{label}: unknown fields or non-object')


def _unique(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f'duplicate profile field: {key}')
        data[key] = value
    return data


def load_profile(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'non-finite number: {x}')))


def _control(data, assertion):
    if type(data) is not dict:
        raise ValueError('control must be an object')
    kind = data.get('kind')
    table = {
        'clock': (ClockControl, {'name','path','minimum','maximum','governor','min_file','max_file',
                                'governor_file','current_file','verify_current','timeout_s','poll_s','consecutive'}),
        'emc': (EmcControl, {'name','rate_path','lock_path','halt_path','rate_hz'}),
        'fan': (FanControl, {'name','pwm_path','pwm','service','systemctl','timeout_s'}),
        'power_mode': (PowerModeControl, {'name','expected_id','expected_name','command','timeout_s'}),
        'swappiness': (SwappinessControl, {'name','value','path'}),
        'affinity': (AffinityControl, {'name','cpus'}),
    }
    if kind not in table:
        raise ValueError(f'unsupported control kind: {kind}')
    constructor, allowed = table[kind]
    _fields(data, allowed | {'kind'}, 'control')
    kwargs = {key: value for key, value in data.items() if key != 'kind'}
    # Public profiles expose known writable controls, not arbitrary file writes.
    if kind == 'clock':
        choices = {'min_file': {'min_freq','scaling_min_freq'},
                   'max_file': {'max_freq','scaling_max_freq'},
                   'governor_file': {'governor','scaling_governor'},
                   'current_file': {'cur_freq','scaling_cur_freq','cpuinfo_cur_freq'}}
        for key, values in choices.items():
            if key in kwargs and kwargs[key] not in values:
                raise ValueError(f'unsupported clock node: {key}')
    for key in ('path','pwm_path','rate_path','lock_path','halt_path'):
        if key not in kwargs:
            continue
        value = kwargs[key]
        if type(value) is not str:
            raise ValueError('profile paths must be strings')
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('profile paths must be absolute without parent traversal')
        if 'thermal' in path.parts or any(part.startswith('thermal_zone') for part in path.parts):
            raise ValueError('profiles may not write thermal protection controls')
    if kind == 'swappiness' and 'path' in kwargs and kwargs['path'] != '/proc/sys/vm/swappiness':
        raise ValueError('swappiness profiles must target /proc/sys/vm/swappiness')
    if kind == 'fan' and not re.fullmatch(r'pwm[1-9][0-9]*', Path(kwargs.get('pwm_path', '')).name):
        raise ValueError('fan profiles must target a pwmN node')
    if kind == 'emc':
        for key, leaf in (('rate_path','rate'),('lock_path','mrq_rate_locked'),('halt_path','bwmgr_halt')):
            if key in kwargs and Path(kwargs[key]).name != leaf:
                raise ValueError(f'unsupported EMC control node: {key}')
    return constructor(**kwargs, assert_held=assertion)


def environment_from_profile(profile, *, output_dir=None, warmup=None, synchronize=None,
                             lock=None, borrow_lock=False):
    """Build a scope without applying it. Thresholds are supplied by the caller.

    An existing lock may be borrowed in-process. External self-locking programs
    require an observation-only profile to avoid nested hardware lock ownership.
    """
    data = load_profile(profile) if isinstance(profile, (str, Path)) else profile
    _fields(data, {'schema','controls','probes','rules','session','lock_path'}, 'profile')
    if data.get('schema') != 'experiment-control.profile.v1':
        raise ValueError('unsupported experiment environment profile schema')
    for name in ('controls','probes','rules'):
        if type(data.get(name, [])) is not list:
            raise ValueError(f'{name} must be a list')
    session = data.get('session', {})
    _fields(session, {'sample_interval_s','ready_timeout_s','stable_for_s','warmup_runs','on_violation'}, 'session')
    selected_lock = lock
    if selected_lock is None and (data.get('controls') or 'lock_path' in data):
        lock_path = data.get('lock_path', COMMON_LOCK)
        if type(lock_path) is not str or not Path(lock_path).is_absolute() or '..' in Path(lock_path).parts:
            raise ValueError('lock_path must be absolute without parent traversal')
        selected_lock = FileLock(lock_path)
    if lock is not None and 'lock_path' in data:
        raise ValueError('use either an injected lock or profile lock_path')
    assertion = selected_lock.assert_held if selected_lock is not None else None
    controls = [_control(item, assertion) for item in data.get('controls', [])]
    probes = []
    for item in data.get('probes', []):
        _fields(item, {'kind','name','sys_root','proc_root','gpu_path','emc_path','extra_metrics'}, 'probe')
        if item.get('kind') != 'linux':
            raise ValueError('unsupported probe kind')
        probes.append(LinuxProbe(**{k: v for k, v in item.items() if k != 'kind'}))
    rules = []
    for item in data.get('rules', []):
        _fields(item, {'probe','metric','minimum','maximum','expected','max_increase','max_span','required','phases'}, 'rule')
        kwargs = dict(item)
        if 'phases' in kwargs:
            if type(kwargs['phases']) is not list:
                raise ValueError('rule phases must be a list')
            kwargs['phases'] = tuple(kwargs['phases'])
        rules.append(MetricRule(**kwargs))
    return Environment(controls=controls, probes=probes, rules=rules, output_dir=output_dir,
                       lock=selected_lock, borrow_lock=borrow_lock, warmup=warmup,
                       synchronize=synchronize, metadata={'profile': data}, **session)
