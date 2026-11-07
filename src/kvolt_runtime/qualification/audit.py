"""Independent M4 audits of recorded results, without importing GPU libraries.

Energy is recalculated from integer mV/mA and timestamps using exact fractions;
the Runtime's floating-point integration routine is deliberately not reused.
"""
import csv
from dataclasses import asdict
from fractions import Fraction
import math
from pathlib import Path

from .contracts import canonical_json, strict_json, file_sha256, value_sha256

RAILS = ('VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0')
EOS = {128001, 128008, 128009}
FREQUENCIES = {408, 510, 612, 714, 816, 918, 1020, 1122}
GROUPS = {8, 16, 32, 64, 128}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return strict_json(Path(path).read_text(encoding='utf-8'))


def artifact(root, relative):
    require(isinstance(relative, str) and relative and '\\' not in relative
            and all(part not in ('', '.', '..') for part in relative.split('/')), 'unsafe result artifact path')
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'unsafe result artifact path')
    candidate = Path(root) / path
    require(not any(part.is_symlink() for part in (candidate, *candidate.parents)), 'symlink result artifact')
    resolved = candidate.resolve()
    require(resolved.is_relative_to(Path(root).resolve()), 'result artifact escapes session')
    require(resolved.is_file(), 'missing result artifact')
    return resolved


def audit_restoration(before, after, *, external=False):
    """Compare fixed controls, respecting the preparation helper's DVFS rules."""
    if not external:
        require(before['verified'] is True and after['verified'] is True, 'GPU restoration unverified')
        require(before['identity'] == after['identity'], 'restoration changed device/boot identity')
        require(all(before['observed'][key] == after['observed'][key]
                    for key in ('governor', 'min_hz', 'max_hz')), 'GPU settings not restored')
    else:
        require(before['cpu'] == after['cpu'], 'CPU min/max/governor restoration mismatch')
        require(before['mode'] == after['mode'] and before['tpc'] == after['tpc'], 'MAXN/TPC changed')
        for key in ('lock', 'halt'):
            require(before['emc'][key] == after['emc'][key], 'EMC control restoration mismatch')
        if before['emc']['lock'] == 1:
            require(before['emc']['rate'] == after['emc']['rate'], 'locked EMC rate restoration mismatch')
        require(before['fan_service'] == after['fan_service']
                and before['fan_policy'] == after['fan_policy'], 'fan service/policy restoration mismatch')
        if before['fan_service'] == 'inactive':
            require(before['fan_pwm'] == after['fan_pwm'], 'inactive fan PWM restoration mismatch')
    return {'status': 'passed', 'fixed_state_restored': True}


def audit_prepared(receipt):
    observed = receipt['observed']
    require(receipt['verified'] is True and len(observed['cpu_policies']) == 3
            and len({row['path'] for row in observed['cpu_policies']}) == 3
            and all(row['min_khz'] == row['max_khz'] == 2201600 for row in observed['cpu_policies'])
            and observed['emc_rate_hz'] == observed['emc_cap_hz'] == observed['emc_max_rate_hz'] == 3199000000
            and observed['emc_locked'] == observed['emc_halted'] == 1
            and observed['maxn']['name'] == 'MAXN' and observed['maxn']['mode_id'] == 0
            and observed['tpc'] == '0' and observed['fan_pwm'] == 255
            and observed['fan_policy'] == 'user_space' and observed['fan_service'] == 'inactive',
            'independent device preparation readback mismatch')


def audit_native(provenance, binding, *, raw_prefill):
    require(provenance['native_profile'] == binding['native_profile'] == 'agx2', 'native profile mismatch')
    require(type(provenance['model_instances']) is int and provenance['model_instances'] == 1
            and provenance['raw_prefill_adapter'] is raw_prefill,
            'native model instance/attention mismatch')
    require(provenance['versions'] == binding['backend_versions'], 'native library version mismatch')
    require({key: row['sha256'] for key, row in provenance['model_files'].items()}
            == binding['model_files'], 'native model/tokenizer bytes mismatch')
    for relative, digest in binding['backend_sources'].items():
        require(provenance['sources'][relative]['sha256'] == digest, 'native source bytes mismatch')
    require({name: row['sha256'] for name, row in provenance['binaries'].items()}
            == {row['module']: row['sha256'] for row in binding['native_binaries'].values()},
            'native kernel build mismatch')
    options = provenance['effective_overrides']
    require(options['torch_dtype'] == 'float16' and options['do_sample'] is False
            and options['num_beams'] == 1 and options['use_cache'] is True
            and set(options['eos_token_ids']) == EOS, 'native generation settings mismatch')


def audit_warmup(session):
    warmup = session['warmup']
    require(warmup['policy'] == 'formal' and warmup['formal_coverage_qualified'] is True
            and warmup['coverage_scope'] == 'actual_inputs_natural_reachable_shapes'
            and not warmup['errors'] and warmup['interrupted'] is False, 'formal warmup incomplete')
    require(warmup['restore']['verified'] is True and warmup['prepare']['verified'] is True,
            'formal warmup restoration/preparation')
    receipt = warmup['frequency']
    require(receipt['verified'] is True and all(receipt['observed'][key] == 1122000000
            for key in ('min_hz', 'max_hz', 'cur_hz')), 'formal warmup frequency mismatch')
    contexts = {}
    for result in session['results']:
        request = result['request']
        key = (request['token_ids_sha256'], request['prompt_sha256'], canonical_json(request['metadata']))
        context = contexts.setdefault(key, {'length': request['length'], 'nmax': 1, 'requests': []})
        context['nmax'] = max(context['nmax'], request['nmax'])
        context['requests'].append({'request_id': request['request_id'], 'nmax': request['nmax']})
    expected = {(key, group) for key, row in contexts.items()
                for group in (GROUPS if row['nmax'] > 1 else (None,))}
    seen = set()
    for case in warmup['cases']:
        key = (case['input_token_sha256'], case['prompt_sha256'], canonical_json(case['metadata']))
        identity = (key, case['group'])
        require(identity in expected and identity not in seen, 'warmup input/group coverage mismatch')
        seen.add(identity)
        context = contexts[key]
        require(type(case['length']) is int and type(case['nmax']) is int
                and case['length'] == context['length'] and case['nmax'] == context['nmax']
                and case['requests'] == context['requests'], 'warmup frozen context/cap mismatch')
        ids, n, length = case['token_ids'], case['n_output_tokens'], case['length']
        require(case['status'] == 'completed' and case['cache_released'] is True
                and type(n) is int and 1 <= n == len(ids) <= case['nmax'], 'warmup generation incomplete')
        require(all(type(token) is int and 0 <= token <= 128255 for token in ids)
                and not any(token in EOS for token in ids[:-1]), 'warmup invalid tokens/continued EOS')
        require(case['stop_reason'] == ('eos' if ids[-1] in EOS else 'length')
                and (ids[-1] in EOS or n == case['nmax']), 'warmup natural termination mismatch')
        require(all(type(case[field]) is int for field in ('prefill_forwards', 'initial_packs', 'decode_forwards', 'final_cache_length'))
                and case['prefill_forwards'] == 1 and case['initial_packs'] == int(n > 1)
                and case['decode_forwards'] == n-1 and case['final_cache_length'] == length+n-1,
                'warmup forward/packing/cache count mismatch')
        shape = lambda t: {'cache_length': t, 'key_residual_length': t % 128,
                           'value_residual_length': min(t, 128)}
        require(case['initial_pack_shape'] == (shape(length) if n > 1 else None), 'warmup initial pack shape mismatch')
        require(case['decode_shapes'] == [{**shape(length+i), 'result_cache_length': length+i+1}
                                         for i in range(n-1)], 'warmup reached Decode shapes mismatch')
        require(warmup['started_ns'] <= case['started_ns'] < case['ended_ns'] <= warmup['ended_ns'],
                'warmup case timing outside ledger')
    require(seen == expected, 'warmup missing input/group coverage')
    require(warmup['ended_ns'] < session['results'][0]['timing']['t0_ns'], 'warmup overlaps request timing')
    return {'status': 'passed', 'cases': len(seen)}


def integrate_trace(rows, start, end, max_gap_ns):
    """Return exact J per rail; require strict outside brackets, even for N=1."""
    require(type(start) is int and type(end) is int and 0 <= start <= end, 'invalid energy interval')
    require(type(max_gap_ns) is int and max_gap_ns > 0, 'invalid energy gap')
    require(rows and all(row.get('rail') in RAILS for row in rows), 'unknown or empty rails')
    result = {}
    for rail in RAILS:
        points = []
        for row in rows:
            if row['rail'] != rail:
                continue
            a, b, t = (row.get(key) for key in ('read_start_ns', 'read_end_ns', 'timestamp_ns'))
            require(all(type(x) is int and x >= 0 for x in (a, b, t)) and a <= b and t == (a+b)//2,
                    'invalid power read window')
            require(row.get('error') is None, 'power sample read error')
            mv, ma = row.get('voltage_mv'), row.get('current_ma')
            require(type(mv) is int and type(ma) is int, 'raw sensor units missing')
            power = Fraction(mv * ma, 1_000_000)
            require(type(row.get('power_w')) in (int, float) and math.isfinite(row['power_w'])
                    and row['power_w'] == float(power), 'recorded power differs from mV/mA')
            require(not points or points[-1][0] < t, 'nonincreasing sensor samples')
            points.append((t, power))
        require(points and points[0][0] < start and end < points[-1][0], 'missing strict power bracket')
        energy = Fraction(0)
        for (a, pa), (b, pb) in zip(points, points[1:]):
            lo, hi = max(a, start), min(b, end)
            if lo >= hi:
                continue
            require(b-a <= max_gap_ns, 'power gap exceeds profile')
            plo = pa + (pb-pa) * Fraction(lo-a, b-a)
            phi = pa + (pb-pa) * Fraction(hi-a, b-a)
            energy += (plo+phi) * Fraction(hi-lo, 2_000_000_000)
        result[rail] = energy
    return result


def close_energy(actual, exact):
    expected = float(exact)
    require(type(actual) in (int, float) and math.isfinite(actual), 'invalid recorded energy')
    # Accumulation error grows with trace length. This bound is numerical only,
    # never an accuracy claim about the physical sensor or system energy.
    require(abs(actual-expected) <= max(1e-10, abs(expected)*1e-12), 'independent energy mismatch')


def audit_policy(result, repository=None):
    pc, dc = result['decisions']['prefill'], result['decisions']['decode']
    timing, request = result['timing'], result['request']
    candidates = [row['prediction'] for row in pc['candidates']]
    require(all(row['length'] == request['length'] for row in candidates), 'Prefill LUT length mismatch')
    require(len(candidates) == 8 and {row['frequency_mhz'] for row in candidates} == FREQUENCIES,
            'prefill candidate coverage')
    require(pc['elapsed_ns'] == pc['tp_ns']-timing['t0_ns'], 'PC elapsed boundary')
    require(pc['budget_s'] == pc['target_s']-pc['elapsed_ns']/1e9, 'PC budget')
    if repository is not None:
        require(canonical_json(candidates) == canonical_json([asdict(row) for row in repository.prefill(request['length'])])
                and pc['target_s'] == repository.prefill_target(request['length']), 'Prefill predictions/target differ from sealed LUT')
    active = [row for row in candidates if row['ttft_s'] <= pc['budget_s']]
    key = (lambda row: (row['energy_j'], row['ttft_s'], row['frequency_mhz'])) if active else (
        lambda row: (row['ttft_s'], row['energy_j'], row['frequency_mhz']))
    require(pc['selected'] == min(active or candidates, key=key), 'PC raw-energy/tie selection')
    require(pc['selection_path'] == ('energy_min' if active else 'latency_fallback'), 'PC branch')
    prefill_receipt = result['device']['prefill']
    require(pc['tp_ns'] <= prefill_receipt['start_ns'] <= prefill_receipt['end_ns'] < timing['t1_ns'],
            'prefill clock application outside measured PC/Prefill interval')
    if len(result['generation']['token_ids']) == 1:
        require(dc['status'] == 'skipped' and dc['selected'] is None and not dc['candidates'], 'DC skip')
        require(result['device']['decode'] is None, 'unexpected Decode clock application')
    else:
        require(dc['t1_ns'] == timing['t1_ns'] and dc['elapsed_ns'] == dc['td_ns']-timing['t1_ns'], 'DC elapsed boundary')
        require(dc['budget_s'] == dc['target_s']-(dc['elapsed_ns']/1e9)/(request['nmax']-1), 'DC budget')
        q = {row['group']: row['score'] for row in dc['quality']}
        require(len(dc['quality']) == 5 and set(q) == GROUPS, 'quality candidate coverage')
        require(dc['knee']['status'] in ('found', 'not_found', 'error_recovered')
                and dc['knee']['groups'] == sorted(set(dc['knee']['groups']))
                and set(dc['knee']['groups']) <= GROUPS
                and bool(dc['knee']['groups']) == (dc['knee']['status'] == 'found'), 'invalid Knee status/groups')
        candidates = [row['prediction'] for row in dc['candidates']]
        require(all(row['length'] == request['length'] for row in candidates), 'Decode LUT length mismatch')
        if repository is not None:
            require(canonical_json(candidates) == canonical_json([asdict(row) for row in repository.decode(request['length'])])
                    and canonical_json(dc['quality']) == canonical_json([asdict(row) for row in repository.quality(request['length'])])
                    and dc['target_s'] == repository.decode_target(request['length']), 'Decode predictions/quality/target differ from sealed LUT')
        require(len(candidates) == 40 and {(row['group'], row['frequency_mhz']) for row in candidates}
                == {(g, f) for g in GROUPS for f in FREQUENCIES},
                'Decode candidate coverage')
        groups = dc['knee']['groups'] if dc['knee']['status'] == 'found' else q
        active = [row for row in candidates if row['group'] in groups and row['tpot_s'] <= dc['budget_s']]
        key = (lambda row: (row['energy_j_per_token'], -q[row['group']], row['tpot_s'], row['group'], row['frequency_mhz'])) if active else (
            lambda row: (row['tpot_s'], -q[row['group']], row['energy_j_per_token'], row['group'], row['frequency_mhz']))
        require(dc['selected'] == min(active or candidates, key=key), 'DC raw-energy/tie selection')
        require(dc['selection_path'] == ('energy_min' if active else 'latency_fallback'), 'DC branch')
        receipt = result['device']['decode']
        require(dc['td_ns'] <= receipt['start_ns'] <= receipt['end_ns'] < result['generation']['token_ready_ns'][1],
                'Decode clock application boundary')
    for phase in ('prefill', 'decode'):
        selection = result['decisions'][phase]['selected']
        if selection is None:
            continue
        receipt = result['device'][phase]
        observed, hz = receipt['observed'], selection['frequency_mhz']*1_000_000
        require(receipt['verified'] is True and observed['min_hz'] == observed['max_hz'] == observed['cur_hz'] == hz,
                'unverified selected GPU frequency')


def audit_session(path, *, require_formal_warmup=False, binding=None, expected_requests=None, repository=None):
    path = Path(path)
    path = artifact(path.parent, path.name)
    root, session = path.parent, read_json(path)
    require(session['status'] == 'completed' and session['exit_code'] == 0 and not session['errors'], 'session failed')
    require(type(session['backend']['model_instances']) is int and session['backend']['model_instances'] == 1, 'model instance count')
    require(session['collect_power'] is True, 'formal verification requires actual collected power')
    require(session['device_preflight']['verified'] is True, 'device preparation unverified')
    audit_prepared(session['device_preflight'])
    results, raw_rows = session['results'], 0
    require(results, 'empty completed session')
    if binding is not None:
        provenance, runtime = session['provenance'], session['provenance']['runtime']
        require(runtime['source_worktree_dirty'] is False
                and runtime['source_sha256'] == binding['runtime_source_sha256']
                and runtime['python'] == binding['python'] and runtime['platform'] == binding['platform']
                and runtime['required_policy_versions'] == binding['policy_versions'], 'runtime source/environment mismatch')
        require(runtime['installed_distribution_versions'] == binding['policy_versions'], 'scientific distribution versions mismatch')
        require(provenance['device_profile'] == binding['device_profile']
                and provenance['power_profile'] == binding['power_profile'], 'device/collector profile mismatch')
        require(all(provenance['lut'][key] == value for key, value in binding['lut'].items()), 'LUT provenance mismatch')
        require(session['device_preflight']['identity'] == binding['identity'], 'device/boot mismatch')
        audit_native(session['backend'], binding, raw_prefill=True)
    if expected_requests is not None:
        require(canonical_json([row['request'] for row in results]) == canonical_json(expected_requests), 'frozen request sequence changed')
    if require_formal_warmup:
        audit_warmup(session)
    for index, result in enumerate(results):
        request, generation, timing, energy = (result[key] for key in ('request', 'generation', 'timing', 'energy'))
        ids, times = generation['token_ids'], generation['token_ready_ns']
        require(all(type(generation[key]) is int for key in ('n_output_tokens', 'prefill_forwards',
                'initial_packs', 'decode_forwards', 'final_cache_length')), 'invalid generation counter types')
        require(all(type(value) is int and value >= 0 for value in (*times, timing['t0_ns'], timing['t1_ns'], timing['tN_ns'])), 'invalid timing counter types')
        require(type(request['index']) is int and request['index'] == index and type(request['length']) is int
                and 1024 <= request['length'] <= 8192 and len(request['token_ids']) == request['length']
                and value_sha256(request['token_ids']) == request['token_ids_sha256']
                and type(request['nmax']) is int and 1 <= request['nmax'] <= 512, 'request identity/length/cap mismatch')
        require(all(type(token) is int and 0 <= token <= 128255 for token in (*ids, *request['token_ids'])),
                'invalid original token IDs')
        n = len(ids)
        require(generation['status'] == 'completed' and 1 <= n <= request['nmax'], 'incomplete generation')
        require(n == generation['n_output_tokens'] == len(times), 'token count mismatch')
        require(generation['prefill_forwards'] == 1 and generation['initial_packs'] == int(n > 1)
                and generation['decode_forwards'] == n-1, 'forward/packing count mismatch')
        require(generation['final_cache_length'] == request['length']+n-1, 'final cache length mismatch')
        require(not any(token in EOS for token in ids[:-1]), 'generation continued after EOS')
        require((generation['stop_reason'] == 'eos') == (ids[-1] in EOS), 'EOS stop reason mismatch')
        require(ids[-1] in EOS or n == request['nmax'], 'premature length stop')
        require(timing['valid'] is True and timing['t0_ns'] < times[0]
                and all(a < b for a, b in zip(times, times[1:])), 'invalid token timing')
        require((timing['t1_ns'], timing['tN_ns']) == (times[0], times[-1]), 'timing boundary mismatch')
        require(timing['ttft_s'] == (times[0]-timing['t0_ns'])/1e9, 'TTFT normalization mismatch')
        if n == 1:
            require(timing['mean_tpot_s'] is None and energy['decode_energy_j'] is None
                    and energy['decode_j_per_token'] is None, 'N=1 Decode values must remain null')
            require(all(energy['rails'][rail]['decode_energy_j'] is None
                        and energy['rails'][rail]['decode_j_per_token'] is None for rail in RAILS),
                    'N=1 per-rail Decode values must remain null')
        else:
            expected = float(Fraction(times[-1]-times[0], (n-1)*1_000_000_000))
            require(abs(timing['mean_tpot_s']-expected) <= 3*math.ulp(expected), 'TPOT N-1 normalization mismatch')
        require(not result['errors'] and result['cleanup']['status'] == 'completed'
                and result['cleanup']['cache_released'] is True and result['cleanup']['gpu_restored'] is True,
                'request cleanup unverified')
        audit_restoration(read_json(artifact(root, 'device.before.json')), result['device']['restore'])
        if binding is not None:
            for receipt in result['device'].values():
                if receipt is not None:
                    require(receipt['identity'] == binding['identity'] and receipt['profile'] == binding['device_profile'],
                            'request device receipt profile/identity mismatch')
            require(result['power']['profile'] == binding['power_profile'], 'actual collector profile mismatch')
        audit_policy(result, repository)
        require(energy['valid'] is True and not result['power']['errors'], 'invalid collected power')
        power = result['power']
        raw_path = artifact(root, power['raw_path'])
        require(file_sha256(raw_path) == power['raw_sha256'], 'raw trace hash mismatch')
        csv_path = artifact(root, power['csv_path'])
        require(file_sha256(csv_path) == power['csv_sha256'], 'CSV hash mismatch')
        rows = [strict_json(line) for line in raw_path.read_text().splitlines()]
        fields = ('rail', 'timestamp_ns', 'read_start_ns', 'read_end_ns', 'current_ma', 'voltage_mv', 'power_w', 'error')
        with csv_path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == list(fields), 'CSV fields differ from raw schema')
            csv_rows = list(reader)
        require(csv_rows == [{key: '' if row.get(key) is None else str(row[key]) for key in fields} for row in rows],
                'CSV rows differ from original raw power')
        raw_rows += len(rows)
        boundaries = [(timing['t0_ns'], times[0], 'prefill_energy_j'),
                      (timing['t0_ns'], times[-1], 'total_energy_j')]
        if n > 1:
            boundaries.append((times[0], times[-1], 'decode_energy_j'))
        for start, end, name in boundaries:
            exact = integrate_trace(rows, start, end, power['profile']['max_gap_ns'])
            close_energy(energy[name], sum(exact.values()))
            for rail in RAILS:
                close_energy(energy['rails'][rail][name], exact[rail])
            if name == 'decode_energy_j':
                close_energy(energy['decode_j_per_token'], sum(exact.values())/(n-1))
                for rail in RAILS:
                    close_energy(energy['rails'][rail]['decode_j_per_token'], exact[rail]/(n-1))
        require(file_sha256(artifact(root, request['input_path'])) == request['prompt_sha256'], 'input hash mismatch')
        for field, key in (('generation_path', 'generation'), ('result_path', None), ('request_path', 'request')):
            require(read_json(artifact(root, result['paths'][field])) == (result if key is None else result[key]),
                    'saved result differs from session')
    require(all(type(value) is int and value >= 0 for value in session['counts'].values()), 'invalid aggregate counter types')
    require(session['counts']['generation_completed'] == session['counts']['timing_valid']
            == session['counts']['energy_valid'] == len(results), 'session aggregate counts mismatch')
    require(all(session['counts'][key] == 0 for key in ('generation_failed', 'generation_interrupted',
            'not_run', 'energy_not_collected', 'request_cleanup_failed')), 'unexpected failed/skipped aggregate counts')
    return {'status': 'passed', 'requests': len(results), 'raw_power_rows': raw_rows,
            'source_commit': session['provenance']['runtime'].get('source_commit'),
            'slo_violations_preserved': sum(result['slo']['ttft_met'] is False
                                         or result['slo']['mean_tpot_met'] is False for result in results),
            'sensor_response_validated': False}


def audit_device(report):
    require(not report['errors'] and report['restored'] is True and report['restore']['verified'] is True,
            'device transition/restoration failed')
    transitions = report['transitions']
    audit_prepared(report['prepared'])
    require(report['common_start']['verified'] is True, 'common start unverified')
    require(len(transitions) == 32, 'incomplete eight-frequency round trip')
    expected = (408, 510, 612, 714, 816, 918, 1020, 1122)
    for index, frequency in enumerate((*expected, *reversed(expected))):
        for receipt in transitions[2*index:2*index+2]:
            observed, hz = receipt['observed'], frequency*1_000_000
            require(receipt['verified'] is True and observed['min_hz'] == observed['max_hz'] == observed['cur_hz'] == hz,
                    'device transition readback mismatch')
            require(observed['governor'] == 'nvhost_podgov' and receipt['target_hz'] == hz
                    and receipt['identity'] == report['capture']['identity'], 'device transition identity/governor mismatch')
        require(transitions[2*index+1]['changed'] is False, 'identical-frequency setter was not skipped')
        require(not transitions[2*index+1]['write_attempts'] and not transitions[2*index+1]['changes'], 'no-op still attempted a setter')
    old, restored = report['capture']['observed'], report['restore']['observed']
    require(all(old[key] == restored[key] for key in ('governor', 'min_hz', 'max_hz')), 'GPU settings not restored')
    audit_restoration(report['capture'], report['restore'])
    return {'status': 'passed', 'receipts': len(transitions), 'restored': True}
