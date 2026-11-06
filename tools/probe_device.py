#!/usr/bin/env python3
"""Read-only discovery/probe, or explicit opt-in GPU transition qualification.

No action sets CPU, EMC, power mode, fan, service state, or file permissions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kvolt_runtime.device import COMMON_LOCK, DeviceController, DeviceProfile, HardwareLock, SysfsIO
from kvolt_runtime.device.controller import FREQUENCIES_MHZ
from kvolt_runtime.execution.contracts import error_record


def _one(candidates, label):
    found = {path.resolve(): path for path in candidates if path.exists()}
    if len(found) != 1:
        raise ValueError(f'{label}: require exactly one existing candidate, found {len(found)}')
    return str(next(iter(found.values())))


def discover(args):
    """Propose explicit paths and report observations; never call setters."""
    io = SysfsIO()
    gpu = _one([Path('/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'),
                Path('/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu')], 'GPU devfreq')
    serial_path = Path('/proc/device-tree/serial-number')
    serial = io.read(serial_path)
    if not serial:
        raise ValueError('physical board serial is empty')
    fans = [path / 'pwm1' for path in Path('/sys/class/hwmon').glob('hwmon*')
            if (path / 'name').exists() and io.read(path / 'name') == 'pwmfan']
    policies = [path / 'policy' for path in Path('/sys/class/thermal').glob('thermal_zone*')
                if (path / 'type').exists() and io.read(path / 'type').lower() in ('tj', 'tj-thermal', 'tj-therm')]
    emc = Path('/sys/kernel/debug/bpmp/debug/clk/emc')
    data = {
        'schema_version': 1, 'name': args.name, 'qualification': 'provisional', 'evidence': args.evidence,
        'gpu_path': gpu, 'identity_path': str(serial_path),
        'identity_expected': hashlib.sha256(serial.encode('utf-8')).hexdigest(),
        'boot_id_path': '/proc/sys/kernel/random/boot_id',
        'cpu_policy_paths': [str(path) for path in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*'))],
        'emc_rate_path': str(emc / 'rate'), 'emc_lock_path': str(emc / 'mrq_rate_locked'),
        'emc_max_rate_path': str(emc / 'max_rate'),
        'emc_cap_path': '/sys/kernel/nvpmodel_clk_cap/emc',
        'emc_halt_path': '/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt',
        'maxn_command': _one([Path('/usr/sbin/nvpmodel'), Path('/usr/bin/nvpmodel')], 'nvpmodel'),
        'maxn_mode_id': 0,
        'tpc_path': _one([Path('/sys/devices/platform/bus@0/17000000.gpu/tpc_pg_mask'),
                          Path('/sys/devices/platform/17000000.gpu/tpc_pg_mask')], 'TPC mask'),
        'tpc_expected': '0', 'fan_path': _one(fans, 'pwmfan PWM'),
        'fan_policy_path': _one(policies, 'Tj thermal policy'),
        'systemctl_command': _one([Path('/usr/bin/systemctl'), Path('/bin/systemctl')], 'systemctl'),
        'timeout_ns': args.timeout_ns, 'poll_ns': args.poll_ns,
        'consecutive_matches': args.matches, 'query_timeout_ns': args.query_timeout_ns,
    }
    profile = DeviceProfile.from_dict(data)
    return profile


def run_probe(controller):
    result = {'capture': controller.capture(), 'errors': [], 'readonly': True}
    try:
        result['prepared'] = controller.verify_prepared()
    except Exception as error:
        result['errors'].append(error_record(error))
    return result


def qualify(controller):
    """Explicit GPU-only transitions with finally restoration, never degraded continuation."""
    result = {'capture': controller.capture(), 'errors': [], 'transitions': [], 'readonly': False,
              'qualification': 'development_device_probe_not_M4', 'restored': False}
    try:
        result['prepared'] = controller.verify_prepared()
        result['common_start'] = controller.prepare_request()
        for frequency in (*FREQUENCIES_MHZ, *reversed(FREQUENCIES_MHZ)):
            result['transitions'].append(controller.apply(frequency))
            result['transitions'].append(controller.apply(frequency))
    except Exception as error:
        result['errors'].append(error_record(error))
    finally:
        try:
            result['restore'] = controller.restore(result['capture'])
            result['restored'] = result['restore']['verified']
        except Exception as error:
            result['errors'].append(error_record(error, stage='device_restore'))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write JSON here as well as stdout')
    sub = parser.add_subparsers(dest='action', required=True)
    discovery = sub.add_parser('discover', help='read-only provisional profile and preparation report')
    discovery.add_argument('--name', required=True)
    discovery.add_argument('--evidence', required=True)
    discovery.add_argument('--timeout-ns', type=int, required=True)
    discovery.add_argument('--poll-ns', type=int, required=True)
    discovery.add_argument('--matches', type=int, required=True)
    discovery.add_argument('--query-timeout-ns', type=int, required=True)
    discovery.add_argument('--profile-output', type=Path, help='save the proposed provisional JSON profile')
    for action in ('probe', 'qualify'):
        command = sub.add_parser(action, help='read-only checks' if action == 'probe' else 'explicitly run GPU transition/restore probe')
        command.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args(argv)
    report = {'action': args.action, 'start_ns': time.monotonic_ns(), 'errors': []}
    try:
        # This process takes the lock itself. Do not wrap it in outer flock.
        with HardwareLock(COMMON_LOCK) as lock:
            profile = discover(args) if args.action == 'discover' else DeviceProfile.from_json(args.profile)
            report['profile'] = profile.as_dict()
            if args.action == 'discover' and args.profile_output:
                args.profile_output.write_text(json.dumps(profile.as_dict(), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            controller = DeviceController(profile, lock=lock)
            report['result'] = qualify(controller) if args.action == 'qualify' else run_probe(controller)
            report['errors'].extend(report['result']['errors'])
    except Exception as error:
        report['errors'].append(error_record(error, stage='device_probe'))
    report['end_ns'] = time.monotonic_ns()
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded, encoding='utf-8')
    print(encoded, end='')
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
