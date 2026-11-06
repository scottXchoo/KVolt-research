"""Host MONOTONIC_NS timing and nonoverlapping INA3221 rail integration.

No prediction enters these measurements. Failed/incomplete requests retain
prefix diagnostics but cannot become normal complete-request observations.
"""
import math

RAILS = ('VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0')
SCOPE = 'INA3221 sum: VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0; not wall power'
ENERGY_FIELDS = ('prefill_energy_j', 'decode_energy_j', 'total_energy_j', 'decode_j_per_token')


def _ns(value):
    return type(value) is int and value >= 0


def timing_metrics(t0_ns, token_ready_ns, *, completed):
    """Use already synchronized token-ready boundaries; never retimestamp.

    ``token_ready_ns`` includes the first token and EOS when generated. Integer
    subtraction precedes conversion to seconds to preserve nanosecond precision.
    """
    tokens = list(token_ready_ns)
    result = dict(valid=False, reason=None, ttft_s=None, mean_tpot_s=None,
                  decode_duration_s=None, t0_ns=t0_ns,
                  t1_ns=tokens[0] if tokens else None, tN_ns=tokens[-1] if tokens else None,
                  token_ready_ns=tokens, n_output_tokens=len(tokens),
                  n_decode_tokens=max(0, len(tokens)-1), prefix=None, decode_reason=None,
                  clock='time.monotonic_ns', unit='s')
    if type(completed) is not bool:
        result['reason'] = 'invalid_completed'; return result
    if not _ns(t0_ns) or any(not _ns(t) for t in tokens):
        result['reason'] = 'invalid_timestamp'; return result
    if not tokens:
        result['reason'] = 'no_output_tokens'; return result
    if tokens[0] <= t0_ns or any(b <= a for a, b in zip(tokens, tokens[1:])):
        result['reason'] = 'nonincreasing_token_time'; return result
    prefix = dict(ttft_s=(tokens[0]-t0_ns)/1e9, mean_tpot_s=None, decode_duration_s=None)
    if len(tokens) > 1:
        prefix['decode_duration_s'] = (tokens[-1]-tokens[0])/1e9
        prefix['mean_tpot_s'] = (tokens[-1]-tokens[0])/(len(tokens)-1)/1e9
    else:
        result['decode_reason'] = 'no_decode_tokens'
    if not completed:
        result.update(reason='incomplete_generation', prefix=prefix); return result
    result.update(prefix, valid=True)
    return result


def _integrate(points, start, end, max_gap):
    """Linear interpolation on each bounding pair; no extrapolation or filling."""
    energy = 0.0
    for left, right in zip(points, points[1:]):
        a, pa = left; b, pb = right
        lo, hi = max(a, start), min(b, end)
        if hi <= lo:
            continue
        if b-a > max_gap:
            raise ValueError('sample_gap_exceeded')
        plo = pa + (pb-pa)*(lo-a)/(b-a)
        phi = pa + (pb-pa)*(hi-a)/(b-a)
        contribution = (plo/2 + phi/2)*((hi-lo)/1e9)
        energy += contribution
        if not all(math.isfinite(v) for v in (plo, phi, contribution, energy)):
            raise ValueError('nonfinite_integration')
    return energy


def energy_metrics(samples, t0_ns, t1_ns, tN_ns, n, *, completed, max_gap_ns):
    """Integrate each rail before summing; signed finite sensor values retained.

    Each rail requires a strictly earlier/later point than the complete interval.
    Failed/missing rows are never silently filtered; invalid traces remain raw.
    ``None`` means collection disabled, while an empty trace means collection failed.
    """
    result = dict(valid=False, reason=None, decode_reason=None, prefix=None,
                  scope=SCOPE, unit='J; J/Decode token', rails={},
                  sensor_response_validated=False, boundary_method='linear interpolation; trapezoid',
                  **dict.fromkeys(ENERGY_FIELDS))
    if samples is None:
        result['reason']='not_collected'; return result
    if (type(completed) is not bool or not _ns(t0_ns) or not _ns(t1_ns) or not _ns(tN_ns)
            or type(n) is not int or n < 1 or not _ns(max_gap_ns) or max_gap_ns == 0
            or not t0_ns < t1_ns <= tN_ns or (n == 1) != (t1_ns == tN_ns)):
        result['reason']='invalid_boundaries'; return result
    rows = list(samples)
    if not rows:
        result['reason']='empty_trace'; return result
    if any(not isinstance(row, dict) or row.get('rail') not in RAILS for row in rows):
        result['reason']='unknown_or_missing_rail'; return result
    values = {}
    try:
        for rail in RAILS:
            points=[]
            for row in rows:
                if row['rail'] != rail:
                    continue
                t, p = row.get('timestamp_ns'), row.get('power_w')
                if (not _ns(t) or type(p) not in (int,float) or not math.isfinite(p)
                        or row.get('error')):
                    raise ValueError(rail+':invalid_sample')
                if 'read_start_ns' in row or 'read_end_ns' in row:
                    a,b=row.get('read_start_ns'),row.get('read_end_ns')
                    if not _ns(a) or not _ns(b) or b < a or t != (a+b)//2:
                        raise ValueError(rail+':invalid_read_window')
                if points and t <= points[-1][0]:
                    raise ValueError(rail+':nonincreasing_sample_time')
                points.append((t,p))
            if not points or points[0][0] >= t0_ns:
                raise ValueError(rail+':missing_left_bracket')
            if points[-1][0] <= tN_ns:
                raise ValueError(rail+':missing_right_bracket')
            prefill=_integrate(points,t0_ns,t1_ns,max_gap_ns)
            decode=_integrate(points,t1_ns,tN_ns,max_gap_ns) if n > 1 else None
            total=_integrate(points,t0_ns,tN_ns,max_gap_ns)
            values[rail]=dict(prefill_energy_j=prefill,decode_energy_j=decode,
                              total_energy_j=total,decode_j_per_token=decode/(n-1) if n > 1 else None)
        sums={key:sum(values[rail][key] for rail in RAILS) for key in ENERGY_FIELDS
              if key not in ('decode_energy_j','decode_j_per_token') or n > 1}
        if any(not math.isfinite(v) for v in sums.values()):
            raise ValueError('nonfinite_rail_sum')
        sums.update({key:None for key in ENERGY_FIELDS if key not in sums})
    except (ValueError, OverflowError) as error:
        result['reason']=str(error); return result
    if n == 1:
        result['decode_reason']='no_decode_tokens'
    if not completed:
        result.update(reason='incomplete_generation',prefix=dict(sums,rails=values)); return result
    result.update(sums,rails=values,valid=True)
    return result
