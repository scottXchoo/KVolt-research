#!/usr/bin/env python3
"""Fixed Decode development experiment; never a production Method execution.

Normal PC/DC decisions are computed afresh, then a separately declared fixed
Decode intervention is applied. Natural EOS and token-ready timing stay intact.
Single observations do not establish statistical speedup or quality equivalence.
"""
import argparse
from dataclasses import asdict, replace
import gc
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ((64, 918), (64, 1122), (128, 1122))
REQUEST_ID = 'm4-cap512-3'


def case_schedule():
    return [(arm, kind, cap) for arm in ARMS
            for kind, cap in (('warmup', 512), ('measured', 2), ('measured', 512))]


def validate_request(spec, original):
    expected = {'request_id': REQUEST_ID, 'length': 8192, 'nmax': 512,
                'prompt_sha256': spec.prompt_sha256, 'token_ids': list(spec.token_ids),
                'token_ids_sha256': spec.token_sha256}
    if spec.request_id != REQUEST_ID or spec.length != 8192 or spec.nmax != 512:
        raise ValueError('expected frozen 8192-token m4-cap512-3 cap512 input')
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError('frozen request fingerprint changed')


class Interrupted(Exception):
    pass


def run_case(spec, arm, *, backend, device, pc, dc, collector, clock, stop):
    """One fresh cache and captured GPU state; seams are CPU-testable.

    Stage wrappers are host brackets, overlap token intervals and must not be
    added to measured TPOT. A cleanup-only failure preserves complete metrics.
    """
    from kvolt_runtime.contracts import generation_stop
    from kvolt_runtime.telemetry import energy_metrics, samples_csv, timing_metrics
    row = {'request_id': spec.request_id, 'nmax': spec.nmax, 'length': spec.length,
           'input_token_sha256': spec.token_sha256, 'prompt_sha256': spec.prompt_sha256,
           'experimental_setting': {'group': arm[0], 'frequency_mhz': arm[1]},
           'normal_decisions': {}, 'status': 'failed', 'generation_completed': False,
           'token_ids': [], 'token_ready_ns': [], 'stages': [], 'errors': [],
           'device': {}, 'cleanup': {}, 'stop_reason': None, 'final_cache_length': None}
    cache = snapshot = prefill = decode = None
    t0 = None

    def boundary():
        if stop.requested:
            raise Interrupted('signal requested: '+str(stop.signal_number))

    def verified(receipt):
        if receipt.get('verified') is not True:
            raise ValueError('unverified GPU operation')
        return receipt

    def stage(name, function, *args):
        trace = {'component': name, 'start_ns': clock(), 'end_ns': None, 'status': 'failed'}
        try:
            result = function(*args)
            trace['status'] = 'completed'
            return result
        finally:
            trace['end_ns'] = clock()
            row['stages'].append(trace)

    def append(token, expected_length):
        previous = row['token_ready_ns'][-1] if row['token_ready_ns'] else t0
        if token.cache is not cache or cache.metadata.length != expected_length:
            raise ValueError('token/cache identity or length mismatch')
        if type(token.ready_ns) is not int or token.ready_ns <= previous:
            raise ValueError('token-ready clock must strictly increase')
        row['token_ids'].append(token.token_id)
        row['token_ready_ns'].append(token.ready_ns)

    def error(exc, stage_name):
        row['errors'].append({'stage': stage_name, 'type': type(exc).__name__, 'message': str(exc)})

    try:
        snapshot = device.capture()
        boundary()
        row['device']['prepare'] = verified(device.prepare_request())
        boundary()
        if collector is not None:
            row['power_start'] = collector.start()
            if not row['power_start'].get('ready') or row['power_start'].get('errors'):
                raise ValueError('collector not ready before t0')
        boundary()
        t0 = clock()
        prefill = pc.choose(spec.length, t0)
        boundary()
        row['device']['prefill'] = verified(device.apply(prefill.selected.frequency_mhz))
        boundary()
        first = stage('prefill_raw', backend.prefill_raw, spec)
        cache = first.cache
        append(first, spec.length)
        reason = generation_stop(tuple(row['token_ids']), spec.nmax)
        if reason is None:
            boundary()
            decode = dc.choose(spec.length, spec.nmax, first.token_id, first.ready_ns)
            boundary()
            if decode.selected is None:
                raise ValueError('normal active Decode has no selected setting')
            row['device']['experimental_decode'] = verified(device.apply(arm[1]))
            boundary()
            stage('pack_initial', backend.pack_initial, cache, arm[0])
            boundary()
            while reason is None:
                token = stage('decode_one', backend.decode_one, cache, row['token_ids'][-1])
                append(token, spec.length+len(row['token_ids']))
                reason = generation_stop(tuple(row['token_ids']), spec.nmax)
                if reason is None:
                    boundary()
        row.update(status='completed', generation_completed=True, stop_reason=reason.value)
    except Interrupted as exc:
        row.update(status='interrupted', stop_reason='interrupted')
        error(exc, 'signal')
    except Exception as exc:
        error(exc, 'generation')
    finally:
        times = row['token_ready_ns']
        # Serialization, telemetry, release and restore are independent cleanup
        # operations. Failure of one must never bypass the others.
        try:
            row['normal_decisions'] = {'prefill': asdict(prefill) if prefill else None,
                                       'decode': asdict(decode) if decode else None}
            if cache is not None:
                row['final_cache_length'] = cache.metadata.length
        except Exception as exc:
            error(exc, 'decision_evidence')
        try:
            row['timing'] = timing_metrics(t0, times, completed=row['generation_completed'])
            if row['generation_completed'] and not row['timing']['valid']:
                raise ValueError('invalid complete-request timing')
        except Exception as exc:
            error(exc, 'timing')
            row['timing'] = {'valid': False, 'reason': 'measurement_failure',
                't0_ns': t0, 't1_ns': times[0] if times else None, 'tN_ns': times[-1] if times else None,
                'ttft_s': None, 'mean_tpot_s': None, 'decode_duration_s': None, 'prefix': None}
        row['energy'] = {'valid': False, 'reason': 'not_collected', 'prefix': None,
            'prefill_energy_j': None, 'decode_energy_j': None, 'total_energy_j': None, 'decode_j_per_token': None}
        if collector is not None:
            try:
                receipt = collector.finish(times[-1]) if times else collector.stop()
                row['power'] = {k: v for k, v in receipt.items() if k not in ('samples', 'raw_trace_jsonl')}
                row['traces'] = {'power.csv': samples_csv(receipt.get('samples', [])),
                                 'power.raw.jsonl': receipt.get('raw_trace_jsonl', '')}
                row['energy'] = energy_metrics(receipt.get('samples', []), t0,
                    times[0] if times else None, times[-1] if times else None, len(times),
                    completed=row['generation_completed'], max_gap_ns=receipt['profile']['max_gap_ns'])
                if receipt.get('errors') or (row['generation_completed'] and not row['energy']['valid']):
                    raise ValueError('invalid power measurement')
            except Exception as exc:
                error(exc, 'power_measurement')
                row['energy'].update(valid=False, reason='collector_failure')
                for key in ('prefill_energy_j', 'decode_energy_j', 'total_energy_j', 'decode_j_per_token'):
                    row['energy'][key] = None
            finally:
                try:
                    if collector.stop().get('errors'):
                        raise ValueError('collector stop errors')
                except Exception as exc:
                    error(exc, 'collector_stop')
        try:
            if cache is not None:
                backend.release_request(cache)
            row['cleanup']['cache_released'] = True
        except Exception as exc:
            row['cleanup']['cache_released'] = False
            error(exc, 'cache_release')
        try:
            row['device']['restore'] = verified(device.restore(snapshot)) if snapshot is not None else None
            row['cleanup']['gpu_restored'] = snapshot is not None
        except Exception as exc:
            row['cleanup']['gpu_restored'] = False
            error(exc, 'gpu_restore')
        if row['errors'] and row['status'] != 'interrupted':
            row['status'] = 'failed'
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--latest-dir', type=Path, required=True)
    parser.add_argument('--reference-lut-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        parser.error('explicit administrator invocation required for temporary hardware preparation')
    sys.path.insert(0, str(ROOT/'tools'))
    from profile_tpot import pin_runtime, runtime_modules, source_commit, validate_inputs
    commit = source_commit(ROOT)
    pin_runtime(ROOT)
    from qualify_m4 import hardware_scope, progress, read_json, source_hashes, write
    from kvolt_runtime.backend.model import InferenceBackend
    from kvolt_runtime.cli import _source_metadata
    from kvolt_runtime.config import RuntimeConfig
    from kvolt_runtime.device import DeviceProfile, HardwareLock
    from kvolt_runtime.execution.contracts import StopController
    from kvolt_runtime.execution.inputs import preflight_inputs
    from kvolt_runtime.policy.compact import load_lut_repository
    from kvolt_runtime.policy.decode import DecodeController
    from kvolt_runtime.policy.knee import KneeAdapter
    from kvolt_runtime.policy.prefill import PrefillController
    from kvolt_runtime.qualification.contracts import file_sha256
    from kvolt_runtime.run_cli import _signals, _tokenizer
    from kvolt_runtime.telemetry import INAPowerCollector, PowerProfile
    plan = validate_inputs(args.plan, args.plan_sha256)
    repo = load_lut_repository(args.latest_dir, reference_lut_dir=args.reference_lut_dir)
    tokenizer, tokenizer_source = _tokenizer(plan['model_dir'])
    prepared = preflight_inputs(manifest=Path(plan['manifest']), prompt_file=None, tokenizer=tokenizer, repo=repo)
    spec, = (item.spec for item in prepared.requests if item.spec.request_id == REQUEST_ID)
    original, = (item for item in plan['requests'] if item['request_id'] == REQUEST_ID)
    validate_request(spec, original)
    profile, power = DeviceProfile.from_json(plan['device_profile']), PowerProfile.from_json(plan['power_profile'])
    adapter = KneeAdapter()  # verify actual pinned policy dependencies before hardware changes
    predictions = {(row.group, row.frequency_mhz): asdict(row) for row in repo.decode(spec.length)}
    quality = {row.group: row.score for row in repo.quality(spec.length)}
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status = {'schema_version': 'kvolt-fixed-decode-probe-v1', 'mode': 'development', 'status': 'running',
        'formal_evaluation_run': False, 'formal_performance_qualified': False,
        'command': [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        'runtime_commit': commit, 'source_sha256': source_hashes(), 'runtime_root': str(ROOT),
        'runtime_modules': runtime_modules(ROOT),
        'plan_sha256': file_sha256(args.plan), 'plan_path': str(args.plan.resolve()),
        'plan_source_commit': plan['source_commit'], 'frozen_files': plan['frozen_files'],
        'lut': _source_metadata(repo.provenance), 'latest_dir': str(args.latest_dir.resolve()),
        'reference_lut_dir': str(args.reference_lut_dir.resolve()), 'tokenizer': tokenizer_source,
        'input': original, 'resolved_manifest': prepared.resolved_manifest,
        'device_profile': profile.as_dict(), 'power_profile': read_json(plan['power_profile']),
        'schedule': case_schedule(), 'warmups': [], 'measurements': [], 'errors': [],
        'not_run': [],
        'arm_predictions': [dict(predictions[arm], source_quality_score=quality[arm[0]]) for arm in ARMS],
        'instrumentation': 'host stage brackets overlap token-ready intervals; no extra durations added'}
    write(root/'probe-status.json', status)
    backend, stop, clock = None, StopController(), time.monotonic_ns
    try:
        with _signals(stop), HardwareLock() as lock, hardware_scope(root, '', profile, lock) as device:
            try:
                backend = InferenceBackend(kivi_root=plan['kivi_root'], native_profile='agx2', clock=clock)
                backend.load_session(Path(plan['model_dir']), RuntimeConfig())
                status['backend_provenance'] = backend.provenance
                pc, dc = PrefillController(repo, clock=clock), DecodeController(repo, knee_adapter=adapter, clock=clock)
                for index, (arm, kind, cap) in enumerate(case_schedule()):
                    if stop.requested:
                        raise Interrupted('signal requested before next case: '+str(stop.signal_number))
                    progress(f'fixed experiment {index+1}/9 G{arm[0]} f{arm[1]} {kind} cap{cap}')
                    row = run_case(replace(spec, nmax=cap), arm, backend=backend, device=device,
                        pc=pc, dc=dc, collector=INAPowerCollector(power, clock=clock) if kind == 'measured' else None,
                        clock=clock, stop=stop)
                    row.update(kind=kind, index=index)
                    status['warmups' if kind == 'warmup' else 'measurements'].append(row)
                    for name, content in row.pop('traces', {}).items():
                        trace = root/f'case-{index:02d}'/name
                        trace.parent.mkdir(exist_ok=True)
                        trace.write_text(content, encoding='utf-8')
                        row.setdefault('trace_files', {})[name] = {'path': str(trace), 'sha256': file_sha256(trace)}
                    write(root/f'case-{index:02d}.json', row)
                    write(root/'probe-status.json', status)
                    if row['status'] != 'completed':
                        raise RuntimeError(f'case {index} {row["status"]}; no automatic retry')
            finally:
                backend = None
                gc.collect()
        status['runtime_modules'] = runtime_modules(ROOT)
        status['restoration'] = read_json(root/'restoration-audit.json')
        if len(status['measurements']) != 6 or len(status['warmups']) != 3 or status['restoration']['status'] != 'passed':
            raise ValueError('incomplete experiment or restoration')
        status['same_group_frequency_reproducibility'] = []
        for cap in (2, 512):
            left, right = [row for row in status['measurements']
                           if row['experimental_setting']['group'] == 64 and row['nmax'] == cap]
            status['same_group_frequency_reproducibility'].append({'nmax': cap,
                'tokens_equal': left['token_ids'] == right['token_ids'],
                'stop_reason_equal': left['stop_reason'] == right['stop_reason']})
        status['status'] = 'completed'
    except BaseException as exc:
        status['status'] = 'failed'
        status['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        attempted = {row['index'] for row in status['warmups']+status['measurements']}
        status['not_run'] = [{'index': index, 'arm': arm, 'kind': kind, 'nmax': cap}
            for index, (arm, kind, cap) in enumerate(case_schedule()) if index not in attempted]
        if (root/'restoration-audit.json').exists():
            try:
                status['restoration'] = read_json(root/'restoration-audit.json')
            except Exception as exc:
                status['status'] = 'failed'
                status['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        write(root/'probe-status.json', status)
    progress('fixed development experiment completed; restoration passed')


if __name__ == '__main__':
    main()
