#!/usr/bin/env python3
"""CPU audit of pinned compact predictions against an explicit FP16 target source.

No device is controlled. Fixed clocks are scenario assumptions, never measured
controller costs. Policy execution uses the real PC/DC and pinned Knee adapter;
missing policy dependencies produce explicit unavailable evidence.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

SCHEMA = 'kvolt-latest-tpot-feasibility-v1'
LENGTHS = tuple(range(1024, 8193, 1024))
DEFAULT_ELAPSED_NS = (0, 2_000_000)
DEFAULT_NMAX = (2, 128, 512)


def load_repository(latest_dir, reference_lut_dir):
    # Lazy import permits standalone arithmetic helpers without policy packages.
    from kvolt_runtime.policy.compact import CompactLUTRepository, load_lut_repository
    repo = load_lut_repository(latest_dir, reference_lut_dir=reference_lut_dir)
    if not isinstance(repo, CompactLUTRepository):
        raise ValueError('--latest-dir must contain the pinned A1 or A2 compact LUT')
    return repo


def make_knee_adapter():
    from kvolt_runtime.policy.knee import KneeAdapter
    return KneeAdapter()


def summarize_decode(predictions, target_s):
    """Compare binary64 source values directly, with inclusive time feasibility."""
    minimum = min(predictions, key=lambda p: (p.tpot_s, p.group, p.frequency_mhz))
    maximum = max(predictions, key=lambda p: (p.tpot_s, -p.group, -p.frequency_mhz))
    return {'candidate_count': len(predictions),
            'time_feasible_count': sum(p.tpot_s <= target_s for p in predictions),
            'minimum': asdict(minimum), 'maximum': asdict(maximum),
            'minimum_margin_ms': (target_s - minimum.tpot_s) * 1000,
            'all_infeasible': all(p.tpot_s > target_s for p in predictions)}


def validate_scenarios(elapsed_ns, nmax_values):
    if (not elapsed_ns or not nmax_values
            or any(type(n) is not int or n < 0 for n in elapsed_ns)
            or any(type(n) is not int or not 2 <= n <= 512 for n in nmax_values)):
        raise ValueError('scenario elapsed ns must be nonnegative integers; Nmax must be integers 2..512')
    if len(set(elapsed_ns)) != len(elapsed_ns) or len(set(nmax_values)) != len(nmax_values):
        raise ValueError('duplicate scenario values are unsupported')


def _active_decode_groups(decision, full_predictions):
    """Recover the original active domain, including groups absent from sparse rows."""
    if getattr(decision, 'selection_rule', None) == 'budget_filtered_decode':
        groups = sorted({row.group for row in full_predictions})
        if decision.knee.status == 'found':
            return set(decision.knee.groups)
        if decision.knee.status == 'not_found':
            return {groups[len(groups) // 2]}
        return set(groups)
    # Historical full-grid controllers already recorded every group's activity.
    # Honor their flags, including the earlier all-G normal no-knee policy.
    return {candidate.prediction.group for candidate in decision.candidates if candidate.active_knee}


def policy_scenarios(repo, elapsed_ns=DEFAULT_ELAPSED_NS, nmax_values=DEFAULT_NMAX, *, knee_adapter=None):
    from kvolt_runtime.policy.decode import DecodeController
    from kvolt_runtime.policy.prefill import PrefillController
    validate_scenarios(elapsed_ns, nmax_values)
    adapter = make_knee_adapter() if knee_adapter is None else knee_adapter
    scenarios = []
    origin = 1_000_000_000
    for length in LENGTHS:
        repo.preflight(length)
        # Offline aggregation source only; controller lookup and trace stay sparse.
        full_decode = repo.decode(length)
        for elapsed in elapsed_ns:
            clock = lambda: origin + elapsed
            pc = PrefillController(repo, clock=clock).choose(length, origin)
            for nmax in nmax_values:
                dc = DecodeController(repo, knee_adapter=adapter, clock=clock).choose(length, nmax, 1, origin)
                prefill, decode = asdict(pc), asdict(dc)
                prefill.update(selected_margin_ms=(pc.budget_s - pc.selected.ttft_s) * 1000,
                               time_feasible_count=sum(c.meets_budget for c in pc.candidates))
                active_groups = _active_decode_groups(dc, full_decode)
                decode.update(all40_time_feasible_count=sum(p.tpot_s <= dc.budget_s for p in full_decode),
                              knee_active_count=sum(p.group in active_groups for p in full_decode),
                              knee_active_time_feasible_count=sum(
                                  p.group in active_groups and p.tpot_s <= dc.budget_s for p in full_decode),
                              selected_margin_ms=(dc.budget_s - dc.selected.tpot_s) * 1000,
                              selected_target_margin_ms=(dc.target_s - dc.selected.tpot_s) * 1000)
                # JSON round trip gives lists consistently to library and CLI users.
                scenarios.append(json.loads(json.dumps({
                    'length': length, 'nmax': nmax,
                    'assumed_elapsed_ns': elapsed, 'elapsed_is_measured': False,
                    'prefill': prefill, 'decode': decode}, allow_nan=False)))
    return scenarios


def build_report(latest_dir, reference_lut_dir, *, elapsed_ns=DEFAULT_ELAPSED_NS,
                 nmax_values=DEFAULT_NMAX, run_policy=True):
    from kvolt_runtime.policy.types import PolicyDependencyError
    from kvolt_runtime.config import buffered_tpot_target, resolve_tpot_buffer_ratio
    validate_scenarios(elapsed_ns, nmax_values)
    latest, reference = Path(latest_dir).resolve(), Path(reference_lut_dir).resolve()
    repo = load_repository(latest, reference)
    metadata = json.loads(repo.provenance.metadata_json)
    rows = []
    buffer_ratio = resolve_tpot_buffer_ratio()
    for length in LENGTHS:
        repo.preflight(length)
        raw_reference = repo.decode_target(length)
        effective = buffered_tpot_target(raw_reference, buffer_ratio)
        rows.append({'length': length, 'reference_target_s': raw_reference,
                     'tpot_buffer_ratio': buffer_ratio, 'target_s': effective,
                     'all40': summarize_decode(repo.decode(length), effective)})
    execution = {'status': 'not_requested', 'engine': 'real PrefillController/DecodeController with pinned KneeAdapter',
                 'clock_kind': 'fixed assumed elapsed', 'elapsed_is_measured': False,
                 'assumed_elapsed_ns': list(elapsed_ns), 'nmax_values': list(nmax_values)}
    scenarios = []
    if run_policy:
        try:
            scenarios = policy_scenarios(repo, elapsed_ns, nmax_values)
            execution['status'] = 'completed'
        except PolicyDependencyError as error:
            execution.update(status='unavailable', error_type=type(error).__name__, error=str(error))
    target = metadata.get('target_repository', {})
    return {'schema_version': SCHEMA,
            'context': {'purpose': 'development crossover feasibility audit',
                        'prediction_source_device': repo.provenance.source_device,
                        'target_source_device': metadata.get('target_source_device', target.get('source_device')),
                        'device_mismatch': metadata['cross_device_reference'], 'formal_qualification': False,
                        'runtime_accuracy_validated': False, 'actual_slo_validated': False,
                        'target_policy': 'raw pinned AGX2 FP16 R_d times (1 + global TPOT buffer ratio)',
                        'tpot_buffer_ratio': buffer_ratio,
                        'comparison': 'unrounded binary64 <=; displayed units do not affect comparisons'},
            'inputs': {'latest_dir': str(latest), 'reference_lut_dir': str(reference),
                       'latest_files_sha256': {name: hashlib.sha256((latest / name).read_bytes()).hexdigest()
                                               for name in ('prefill.json', 'decode.json', 'quality.json')},
                       'compact_manifest_sha256': repo.provenance.manifest_sha256,
                       'reference_manifest_sha256': target.get('manifest_sha256'),
                       'reference_files_sha256': target.get('files_sha256'),
                       'provenance': asdict(repo.provenance)},
            'lengths': rows, 'policy_execution': execution, 'policy_scenarios': scenarios}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latest-dir', required=True, type=Path)
    parser.add_argument('--reference-lut-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--elapsed-ns', nargs='+', type=int, default=DEFAULT_ELAPSED_NS)
    parser.add_argument('--nmax', nargs='+', type=int, default=DEFAULT_NMAX)
    parser.add_argument('--arithmetic-only', action='store_true', help='explicitly omit PC/DC; preserve all40 arithmetic')
    args = parser.parse_args(argv)
    try:
        report = build_report(args.latest_dir, args.reference_lut_dir,
                              elapsed_ns=tuple(args.elapsed_ns), nmax_values=tuple(args.nmax),
                              run_policy=not args.arithmetic_only)
        payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    except (ValueError, OSError) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')
    if report['policy_execution']['status'] == 'unavailable':
        print('All40 arithmetic written; PC/DC unavailable: ' + report['policy_execution']['error'], file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
