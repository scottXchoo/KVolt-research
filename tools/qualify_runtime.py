#!/usr/bin/env python3
"""M3 development validation with finally restoration, never formal M4 approval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from kvolt_runtime.device import DeviceController, DeviceProfile, HardwareLock
from kvolt_runtime.execution.contracts import error_record
from kvolt_runtime.qualification.session import ExperimentPreparation
from kvolt_runtime.recording import SessionRecorder
from kvolt_runtime.cli import _runtime_metadata


def _power_profile(path):
    from kvolt_runtime.telemetry import PowerProfile
    return PowerProfile.from_json(path)


def power_probe(profile, recorder):
    from kvolt_runtime.telemetry import INAPowerCollector, energy_metrics, samples_csv
    collector = INAPowerCollector(profile)
    result = {'purpose': 'idle_sensor_plumbing_only_no_inference_or_energy_saving_claim', 'errors': []}
    t0 = t1 = tn = None
    try:
        result['start'] = collector.start()
        if not result['start']['ready']:
            raise RuntimeError('collector was not ready')
        t0 = time.monotonic_ns()
        time.sleep(.15)
        t1 = time.monotonic_ns()
        time.sleep(.15)
        tn = time.monotonic_ns()
        receipt = collector.finish(tn)
        samples = receipt.pop('samples')
        raw = receipt.pop('raw_trace_jsonl')
        recorder.write_text('telemetry/power.raw.jsonl', raw)
        csv = samples_csv(samples)
        recorder.write_text('telemetry/power.csv', csv)
        result.update(receipt=receipt, boundaries={'t0_ns': t0, 't1_ns': t1, 'tN_ns': tn},
                      csv_sha256=hashlib.sha256(csv.encode()).hexdigest(),
                      metrics=energy_metrics(samples, t0, t1, tn, 2, completed=True,
                                             max_gap_ns=profile.max_gap_ns))
        if receipt['errors'] or not result['metrics']['valid']:
            raise RuntimeError('collected power trace is invalid')
    except Exception as error:
        result['errors'].append(error_record(error, stage='power_probe'))
    finally:
        result['stop'] = {key: value for key, value in collector.stop().items()
                          if key not in ('samples', 'raw_trace_jsonl')}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('device', 'power', 'smoke'), required=True)
    parser.add_argument('--device-profile', type=Path, required=True)
    parser.add_argument('--power-profile', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-experiment', action='store_true',
                        help='explicit external CPU/EMC/fan preparation, restored in finally')
    parser.add_argument('run_args', nargs=argparse.REMAINDER, help='smoke: kvolt run arguments after --')
    args = parser.parse_args(argv)
    recorder = SessionRecorder(args.output_dir)
    profile = DeviceProfile.from_json(args.device_profile)
    report = {'suite': args.suite, 'profile': profile.as_dict(), 'started_ns': time.monotonic_ns(),
              'formal_runtime_qualified': False, 'errors': [], 'runtime': _runtime_metadata(),
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result = None
    try:
        with HardwareLock() as lock:
            device = DeviceController(profile, lock=lock)
            original = device.capture()
            recorder.atomic_json('device.before.json', original)
            preparation = ExperimentPreparation(profile, lock=lock)
            try:
                if args.prepare_experiment:
                    report['external_before'] = preparation.capture()
                    recorder.atomic_json('external.before.json', report['external_before'])
                    report['external_prepare'] = preparation.prepare()
                if args.suite == 'device':
                    from probe_device import qualify
                    result = qualify(device)
                    if result['errors'] or not result['restored']:
                        raise RuntimeError('device transition verification failed; see result')
                elif args.suite == 'power':
                    if args.power_profile is None:
                        raise ValueError('--power-profile is required for power probe')
                    result = power_probe(_power_profile(args.power_profile), recorder)
                    if result['errors']:
                        raise RuntimeError('power plumbing verification failed; see result')
                else:
                    from kvolt_runtime.cli import _parser
                    from kvolt_runtime.run_cli import run
                    argv = args.run_args[1:] if args.run_args[:1] == ['--'] else args.run_args
                    parsed = _parser().parse_args(['run', *argv])
                    parsed._original = ['kvolt', 'run', *argv]
                    records = []
                    exit_code = run(parsed, lock=lock, emit=records.append)
                    result = {'exit_code': exit_code, 'stdout_records': records}
                    recorder.atomic_json('run.stdout.json', records)
                    if exit_code:
                        raise RuntimeError(f'run CLI returned {exit_code}')
            except Exception as error:
                report['errors'].append(error_record(error, stage=args.suite))
            finally:
                try:
                    report['gpu_restore'] = device.restore(original)
                    if not report['gpu_restore']['verified']:
                        raise RuntimeError('unverified GPU restoration')
                except Exception as error:
                    report['errors'].append(error_record(error, stage='gpu_restore'))
                report['external_restore'] = preparation.restore()
                report['errors'].extend(report['external_restore']['errors'])
    except Exception as error:
        report['errors'].append(error_record(error, stage='qualification'))
    report.update(result=result, ended_ns=time.monotonic_ns(), status='failed' if report['errors'] else 'passed')
    recorder.atomic_json('qualification.json', report)
    print(json.dumps({'status': report['status'], 'suite': args.suite,
                      'errors': report['errors'], 'path': str(args.output_dir / 'qualification.json')}, allow_nan=False))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
