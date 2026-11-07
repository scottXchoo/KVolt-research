#!/usr/bin/env python3
"""Read-only, stdlib TPOT decomposition for recorded KVolt sessions.

This tool recomputes measured intervals. It never reruns policy, changes a LUT,
adds initial packing to TPOT, or claims missing stage timings were measured.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

GROUPS = (8, 16, 32, 64, 128)
FREQUENCIES = (408, 510, 612, 714, 816, 918, 1020, 1122)
SCHEMA = 'kvolt-tpot-diagnostics-v1'


def _ns(value):
    return type(value) is int and value >= 0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _duration(start, end):
    return (end - start) / 1_000_000_000


def _receipt(receipt):
    if not isinstance(receipt, dict):
        return None
    start, end = receipt.get('start_ns'), receipt.get('end_ns')
    return {**{k: deepcopy(receipt.get(k)) for k in
               ('operation', 'target_hz', 'verified', 'changed', 'before',
                'observed', 'verification', 'write_attempts')},
            'start_ns': start, 'end_ns': end,
            'elapsed_s': _duration(start, end) if _ns(start) and _ns(end) and end >= start else None}


def _candidate_summary(decision):
    candidates = decision.get('candidates') or []
    if not isinstance(candidates, list):
        candidates = []
    pairs, valid = set(), []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        prediction = candidate.get('prediction')
        if not isinstance(prediction, dict):
            continue
        pair = (prediction.get('group'), prediction.get('frequency_mhz'))
        if pair[0] in GROUPS and pair[1] in FREQUENCIES and _finite(prediction.get('tpot_s')) and prediction['tpot_s'] > 0:
            pairs.add(pair)
            valid.append(prediction)
    complete = len(valid) == len(candidates) == len(pairs) == 40
    minimum = min(valid, key=lambda p: p['tpot_s']) if complete else None
    target = decision.get('target_s')
    return {'candidate_count': len(candidates),
            'eligible_count': sum(c.get('eligible') is True for c in candidates if isinstance(c, dict)),
            'meets_budget_count': sum(c.get('meets_budget') is True for c in candidates if isinstance(c, dict)),
            'candidate_coverage': 'complete' if complete else 'incomplete' if candidates else 'not_applicable',
            'all40_min_prediction': deepcopy(minimum),
            'all40_min_exceeds_target': minimum['tpot_s'] > target if minimum and _finite(target) else None}


def _partition(result, times, warnings):
    """Disjoint portions of first gap; uninstrumented remainder stays unnamed."""
    fields = ('dc_preselection_s', 'device_apply_s', 'initial_pack_s', 'first_decode_s', 'uninstrumented_s')
    output = dict.fromkeys(fields)
    if len(times) < 2:
        return output
    lower, upper = times[:2]
    decision = (result.get('decisions') or {}).get('decode') or {}
    device = (result.get('device') or {}).get('decode') or {}
    stages = (result.get('generation') or {}).get('stage_timing') or {}
    intervals = []
    td = decision.get('td_ns')
    if td is not None:
        intervals.append(('dc_preselection_s', lower, td))
    if device:
        intervals.append(('device_apply_s', device.get('start_ns'), device.get('end_ns')))
    for stage in ('initial_pack', 'first_decode'):
        interval = stages.get(stage)
        if isinstance(interval, dict):
            intervals.append((stage + '_s', interval.get('start_ns'), interval.get('end_ns')))
    prior_end = lower
    for name, start, end in intervals:
        # Backend token readiness is stamped before returning to the Runner.
        # Preserve the full call span separately; partition only its overlap.
        if not (_ns(start) and _ns(end) and lower <= start <= end and
                (name == 'first_decode_s' or end <= upper) and start <= upper and start >= prior_end):
            warnings.append('first_gap_partition_invalid:' + name)
            return dict.fromkeys(fields)
        clipped_end = min(end, upper)
        output[name] = _duration(start, clipped_end)
        if clipped_end != end:
            warnings.append('first_decode_span_clipped_to_token_ready')
        prior_end = clipped_end
    measured_ns = sum(min(end, upper) - start for _, start, end in intervals)
    output['uninstrumented_s'] = (upper - lower - measured_ns) / 1_000_000_000
    return output


def analyze_result(result, *, source=None):
    """Return independent arithmetic and recorded selection/device evidence.

    Invalid or partial traces retain diagnostic evidence but publish no measured
    TPOT or SLO classification. N=1 is valid with null Decode measurements.
    """
    request = result.get('request') or {}
    generation = result.get('generation') or {}
    timing = result.get('timing') or {}
    decision = (result.get('decisions') or {}).get('decode') or {}
    errors, warnings = [], []
    times = generation.get('token_ready_ns')
    count = generation.get('n_output_tokens')
    if not isinstance(times, list) or not times:
        errors.append('token_ready_ns_missing_or_empty')
    elif not all(_ns(t) for t in times):
        errors.append('token_ready_ns_not_nonnegative_integers')
    elif any(b <= a for a, b in zip(times, times[1:])):
        errors.append('token_ready_ns_not_strictly_increasing')
    if type(count) is not int or not isinstance(times, list) or count != len(times):
        errors.append('n_output_tokens_mismatch')
    if generation.get('status') != 'completed':
        errors.append('generation_not_completed')
    if timing.get('valid') is False or result.get('errors'):
        errors.append('recorded_technical_invalidity')
    if not errors:
        for key, expected in [('t1_ns', times[0]), ('tN_ns', times[-1]),
                              ('token_ready_ns', times), ('n_output_tokens', count)]:
            supplied = timing.get(key)
            strict = (isinstance(supplied, list) and all(_ns(t) for t in supplied)) if key == 'token_ready_ns' else _ns(supplied)
            if key in timing and (not strict or supplied != expected):
                errors.append('recorded_timing_mismatch:' + key)
        if 't0_ns' in timing and (not _ns(timing['t0_ns']) or timing['t0_ns'] >= times[0]):
            errors.append('t0_ns_invalid')
        for key in ('t1_ns',):
            if decision.get(key) is not None and (not _ns(decision[key]) or decision[key] != times[0]):
                errors.append('recorded_decision_mismatch:' + key)
        duration = _duration(times[0], times[-1]) if count > 1 else None
        mean = duration / (count - 1) if duration is not None else None
        for key, expected in [('decode_duration_s', duration), ('mean_tpot_s', mean)]:
            if key in timing:
                recorded = timing[key]
                if ((expected is None and recorded is not None) or
                    (expected is not None and (not _finite(recorded) or not math.isclose(recorded, expected, rel_tol=1e-12, abs_tol=1e-15)))):
                    errors.append('recorded_timing_mismatch:' + key)
    report = {'source': source, 'request_id': request.get('request_id'),
              'length': request.get('length'), 'nmax': request.get('nmax'),
              'n_output_tokens': count, 'status': 'invalid' if errors else 'valid',
              'errors': errors, 'warnings': warnings,
              'target_s': decision.get('target_s'), 'budget_s': decision.get('budget_s'),
              'selection_path': decision.get('selection_path'),
              'predicted_feasible': decision.get('predicted_feasible'),
              'selected_prediction': deepcopy(decision.get('selected')),
              'stage_timing': deepcopy(generation.get('stage_timing')),
              'device_apply': _receipt((result.get('device') or {}).get('decode')),
              **_candidate_summary(decision),
              **dict.fromkeys(('first_intertoken_s', 'steady_mean_s', 'steady_min_s', 'steady_max_s',
                               'decode_duration_s', 'mean_tpot_s', 'actual_slo_met', 'dc_elapsed_s')),
              'steady_gap_count': 0}
    report['first_gap_partition'] = _partition(result, [], warnings)
    if errors:
        return report
    report['first_gap_partition'] = _partition(result, times, warnings)
    if count > 1:
        report['first_intertoken_s'] = _duration(times[0], times[1])
        report['decode_duration_s'] = _duration(times[0], times[-1])
        report['mean_tpot_s'] = report['decode_duration_s'] / (count - 1)
        gaps_ns = [b - a for a, b in zip(times[1:], times[2:])]
        report['steady_gap_count'] = len(gaps_ns)
        if gaps_ns:
            report.update(steady_mean_s=sum(gaps_ns) / len(gaps_ns) / 1_000_000_000,
                          steady_min_s=min(gaps_ns) / 1_000_000_000,
                          steady_max_s=max(gaps_ns) / 1_000_000_000)
        target = report['target_s']
        if _finite(target) and target > 0:
            report['actual_slo_met'] = report['mean_tpot_s'] <= target
        td = decision.get('td_ns')
        if _ns(td) and times[0] <= td <= times[1]:
            report['dc_elapsed_s'] = _duration(times[0], td)
            if decision.get('elapsed_ns') != td - times[0]:
                warnings.append('dc_elapsed_ns_mismatch')
    return report


def summarize(results):
    return {'request_count': len(results),
            'valid_count': sum(r['status'] == 'valid' for r in results),
            'invalid_count': sum(r['status'] == 'invalid' for r in results),
            'decode_evaluable_count': sum(r['actual_slo_met'] is not None for r in results),
            'actual_tpot_violation_count': sum(r['actual_slo_met'] is False for r in results)}


def analyze_session(session, *, source=None):
    if not isinstance(session, dict) or not isinstance(session.get('results'), list):
        raise ValueError('session must contain a results array')
    results = [analyze_result(r, source=source) for r in session['results']]
    return {'schema_version': SCHEMA, 'source': source, 'session_id': session.get('session_id'),
            'provenance': deepcopy(session.get('provenance')), 'summary': summarize(results), 'results': results}


def _ms(value):
    return '—' if value is None else f'{value * 1000:.3f}'


def format_report(report):
    lines = ['# TPOT diagnostics', '',
             'All timings below are milliseconds. First gap and remaining-gap mean partition the recorded t1→tN interval.',
             'Remaining gaps are descriptive; no stationarity or kernel-only timing is assumed.', '',
             '| Source | Request | L | N | Path | G / MHz | Target | LUT min / selected | First gap | Remaining mean | TPOT | Met |',
             '|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|']
    for r in report['results']:
        selected = r['selected_prediction'] or {}
        minimum = r['all40_min_prediction'] or {}
        source = str(r['source'] or '')
        if source:
            source = Path(source).parent.name
        cells = [source, str(r['request_id']), str(r['length']), str(r['n_output_tokens']),
                 str(r['selection_path']), f"{selected.get('group', '—')} / {selected.get('frequency_mhz', '—')}",
                 _ms(r['target_s']), f"{_ms(minimum.get('tpot_s'))} / {_ms(selected.get('tpot_s'))}",
                 _ms(r['first_intertoken_s']), _ms(r['steady_mean_s']), _ms(r['mean_tpot_s']),
                 'invalid' if r['status'] == 'invalid' else '—' if r['actual_slo_met'] is None else str(r['actual_slo_met'])]
        lines.append('| ' + ' | '.join(c.replace('|', '\\|') for c in cells) + ' |')
    lines.extend(['', 'Summary: ' + json.dumps(report['summary'], sort_keys=True), '',
                  'DC is t1→td (preselection). Device apply comes from its receipt. Missing initial_pack/first_decode timings remain unresolved in first_gap_partition.uninstrumented_s.',
                  'Stage intervals partition the first gap and are never added to TPOT. N=1 has null Decode metrics; N=2 has no remaining gaps.', ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sessions', nargs='+', type=Path, help='recorded session.json files')
    parser.add_argument('--output', type=Path, help='write machine JSON here (default: stdout)')
    parser.add_argument('--report', type=Path, help='write compact Markdown report here')
    args = parser.parse_args(argv)
    try:
        sessions = [analyze_session(json.loads(p.read_text()), source=str(p)) for p in args.sessions]
        results = [r for s in sessions for r in s['results']]
        report = {'schema_version': SCHEMA, 'sessions': sessions, 'summary': summarize(results), 'results': results}
        encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(encoded)
        else:
            sys.stdout.write(encoded)
        if args.report:
            args.report.write_text(format_report(report))
        return 1 if report['summary']['invalid_count'] else 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('TPOT diagnostics: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
