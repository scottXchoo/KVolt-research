"""Known analytical energy and adversarial traces for the independent auditor."""
import copy
import csv
from fractions import Fraction

import pytest

from kvolt_runtime.qualification.audit import (
    RAILS, audit_native, audit_policy, audit_restoration, audit_session, audit_warmup, integrate_trace,
)
from kvolt_runtime.qualification.contracts import canonical_json, file_sha256, value_sha256


def trace():
    # Each rail rises linearly from -1 W to 3 W over two seconds.
    return [dict(rail=rail, timestamp_ns=t, read_start_ns=t, read_end_ns=t,
                 voltage_mv=1000, current_ma=ma, power_w=ma/1000, error=None)
            for rail in RAILS
            for t, ma in ((0, -1000), (1_000_000_000, 1000), (2_000_000_000, 3000))]


def test_exact_boundary_interpolation_preserves_signed_power():
    result = integrate_trace(trace(), 250_000_000, 1_750_000_000, 1_000_000_000)
    assert result == dict.fromkeys(RAILS, Fraction(3, 2))
    assert integrate_trace(trace(), 250_000_000, 500_000_000, 1_000_000_000) == dict.fromkeys(RAILS, Fraction(-1, 16))


@pytest.mark.parametrize('mutation', ['missing', 'read_error', 'wrong_watts', 'wrong_midpoint', 'duplicate_time', 'unknown_rail', 'boolean_power'])
def test_bad_raw_measurements_do_not_become_normal_energy(mutation):
    rows = trace()
    if mutation == 'missing':
        rows = [row for row in rows if row['rail'] != RAILS[0]]
    elif mutation == 'read_error':
        rows[0]['error'] = {'message': 'sensor read failed'}
    elif mutation == 'wrong_watts':
        rows[0]['power_w'] = 0
    elif mutation == 'wrong_midpoint':
        rows[0]['read_end_ns'] = 4
    elif mutation == 'duplicate_time':
        rows.insert(0, dict(rows[0]))
    elif mutation == 'boolean_power':
        rows[1]['power_w'] = True
    else:
        rows[0]['rail'] = 'wall_power'
    with pytest.raises(ValueError):
        integrate_trace(rows, 250_000_000, 1_750_000_000, 1_000_000_000)


def test_exact_boundary_is_not_a_strict_outside_bracket():
    with pytest.raises(ValueError, match='bracket'):
        integrate_trace(trace(), 0, 1_750_000_000, 1_000_000_000)


def test_gap_cannot_be_repaired_by_interpolation():
    with pytest.raises(ValueError, match='gap'):
        integrate_trace(trace(), 250_000_000, 1_750_000_000, 999_999_999)


def partial_prefill_result(*, fallback=False, initially_empty=False):
    # Manually specified screening at 0.1 s and completed lookup at 0.6/0.8 s.
    # 510/612 MHz alone pass screening; final budget may remove one or both.
    t0 = 1_000_000_000
    prefilter = t0 + (1_200_000_000 if initially_empty else 100_000_000)
    elapsed = 1_300_000_000 if initially_empty else (800_000_000 if fallback else 600_000_000)
    tp = t0 + elapsed
    predictions = [{'length': 1024, 'frequency_mhz': f, 'ttft_s': 2.0,
                    'energy_j': None, 'bracket': {'lower': 1024, 'upper': 1024, 'weight': 0.0}}
                   for f in (408, 510, 612, 714, 816, 918, 1020, 1122)]
    predictions[1].update(ttft_s=0.4, energy_j=5.0)
    predictions[2].update(ttft_s=0.4 if initially_empty else 0.5, energy_j=2.0)
    selected = predictions[2 if initially_empty else 1]
    return {'request': {'length': 1024}, 'timing': {'t0_ns': t0, 't1_ns': 3_000_000_000},
            'decisions': {'prefill': {'candidates': [{'prediction': row} for row in predictions],
                'selected': selected, 'prefilter_ns': prefilter, 'tp_ns': tp, 'elapsed_ns': elapsed,
                'target_s': 1.0, 'budget_s': 1.0-elapsed/1e9,
                'selection_path': 'latency_fallback' if fallback or initially_empty else 'energy_min'},
                'decode': {'status': 'skipped', 'selected': None, 'candidates': []}},
            'generation': {'token_ids': [17]},
            'device': {'prefill': {'start_ns': tp+10_000_000, 'end_ns': tp+20_000_000,
                'verified': True, 'observed': dict.fromkeys(('min_hz', 'max_hz', 'cur_hz'),
                    selected['frequency_mhz']*1_000_000)}, 'decode': None}}


@pytest.mark.parametrize('options', [{}, {'fallback': True}, {'initially_empty': True}])
def test_audit_partial_prefill_preserves_final_budget_and_fallback(options):
    audit_policy(partial_prefill_result(**options))


@pytest.mark.parametrize('mutation', ['missing_energy', 'unnecessary_energy', 'nan_energy',
    'clock_before_start', 'clock_after_lookup', 'boolean_clock', 'float_clock', 'missing_clock', 'wrong_screen'])
def test_audit_partial_prefill_rejects_inconsistent_lookup_evidence(mutation):
    result = partial_prefill_result()
    pc = result['decisions']['prefill']
    if mutation == 'missing_energy':
        pc['candidates'][2]['prediction']['energy_j'] = None
    elif mutation == 'unnecessary_energy':
        pc['candidates'][0]['prediction']['energy_j'] = 0.0
    elif mutation == 'nan_energy':
        pc['candidates'][2]['prediction']['energy_j'] = float('nan')
    elif mutation == 'clock_before_start':
        pc['prefilter_ns'] = result['timing']['t0_ns']-1
    elif mutation == 'clock_after_lookup':
        pc['prefilter_ns'] = pc['tp_ns']+1
    elif mutation == 'boolean_clock':
        pc['prefilter_ns'] = True
    elif mutation == 'float_clock':
        pc['prefilter_ns'] = float(pc['prefilter_ns'])
    elif mutation == 'missing_clock':
        del pc['prefilter_ns']
    else:
        pc['prefilter_ns'] = result['timing']['t0_ns']+600_000_000
    with pytest.raises(ValueError):
        audit_policy(result)


@pytest.mark.parametrize('mutation', [None, 'energy', 'time', 'bracket', 'target'])
def test_audit_partial_prefill_checks_queried_and_time_only_rows_against_repository(mutation):
    from kvolt_runtime.policy.mock import MockLUTRepository
    from kvolt_runtime.policy.types import FREQUENCIES_MHZ, PERFORMANCE_LENGTHS

    cells = {f: (2.0, 1.0) for f in FREQUENCIES_MHZ}
    cells.update({510: (0.4, 5.0), 612: (0.5, 2.0)})
    repository = MockLUTRepository(
        prefill={(n, f): cells[f] for n in PERFORMANCE_LENGTHS for f in FREQUENCIES_MHZ},
        prefill_target={n: 1.0 for n in PERFORMANCE_LENGTHS})
    result = partial_prefill_result()
    pc = result['decisions']['prefill']
    if mutation == 'energy':
        pc['candidates'][2]['prediction']['energy_j'] = 3.0
    elif mutation == 'time':
        pc['candidates'][0]['prediction']['ttft_s'] = 1.5
    elif mutation == 'bracket':
        pc['candidates'][0]['prediction']['bracket']['upper'] = 2048
    elif mutation == 'target':
        pc['target_s'] = 1.1
        pc['budget_s'] = pc['target_s']-pc['elapsed_ns']/1e9
    if mutation is None:
        audit_policy(result, repository)
    else:
        with pytest.raises(ValueError, match='sealed LUT'):
            audit_policy(result, repository)


@pytest.fixture
def session_bundle(tmp_path):
    def write(relative, value):
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(value))
    tokens = [7]*1024
    request = {'request_id': 'one', 'attempt': 1, 'index': 0, 'length': 1024, 'nmax': 1,
        'token_ids': tokens, 'token_ids_sha256': value_sha256(tokens), 'metadata': {},
        'input_path': 'input.txt', 'prompt_sha256': ''}
    (tmp_path/'input.txt').write_text('original prompt\r\n')
    request['prompt_sha256'] = file_sha256(tmp_path/'input.txt')
    original = {'verified': True, 'identity': {'boot_id': 'boot', 'physical_id_sha256': 'a'*64},
        'observed': {'governor': 'nvhost_podgov', 'min_hz': 306000000, 'max_hz': 1122000000, 'cur_hz': 408000000}}
    write('device.before.json', original)
    predictions = [{'length': 1024, 'frequency_mhz': frequency, 'ttft_s': 1.0,
                    'energy_j': float(index+1)} for index, frequency in enumerate((408,510,612,714,816,918,1020,1122))]
    result = {'request': request,
        'generation': {'status': 'completed', 'token_ids': [17], 'token_ready_ns': [3_000_000_000],
            'n_output_tokens': 1, 'prefill_forwards': 1, 'initial_packs': 0, 'decode_forwards': 0,
            'final_cache_length': 1024, 'stop_reason': 'length'},
        'timing': {'valid': True, 't0_ns': 1_000_000_000, 't1_ns': 3_000_000_000,
            'tN_ns': 3_000_000_000, 'ttft_s': 2.0, 'mean_tpot_s': None},
        'energy': {'valid': True, 'prefill_energy_j': 6.0, 'total_energy_j': 6.0,
            'decode_energy_j': None, 'decode_j_per_token': None,
            'rails': {rail: {'prefill_energy_j': 2.0, 'total_energy_j': 2.0,
                            'decode_energy_j': None, 'decode_j_per_token': None} for rail in RAILS}},
        'errors': [], 'cleanup': {'status': 'completed', 'cache_released': True, 'gpu_restored': True},
        'decisions': {'prefill': {'candidates': [{'prediction': row} for row in predictions],
            'selected': predictions[0], 'elapsed_ns': 10_000_000, 'tp_ns': 1_010_000_000,
            'target_s': 4.0, 'budget_s': 3.99, 'selection_path': 'energy_min'},
            'decode': {'status': 'skipped', 'selected': None, 'candidates': []}},
        'device': {'prefill': {'verified': True, 'start_ns': 1_020_000_000, 'end_ns': 1_030_000_000,
            'observed': {'min_hz': 408000000, 'max_hz': 408000000, 'cur_hz': 408000000}},
            'decode': None, 'restore': copy.deepcopy(original)},
        'power': {'errors': [], 'raw_path': 'power.jsonl', 'csv_path': 'power.csv',
            'raw_sha256': '', 'csv_sha256': '', 'profile': {'max_gap_ns': 2_000_000_000}},
        'paths': {'generation_path': 'generation.json', 'result_path': 'result.json', 'request_path': 'request.json'},
        'slo': {'ttft_met': False, 'mean_tpot_met': None}}
    rows = [dict(rail=rail, timestamp_ns=t, read_start_ns=t, read_end_ns=t, current_ma=1000,
                 voltage_mv=1000, power_w=1.0, error=None) for rail in RAILS for t in (0, 2_000_000_000, 4_000_000_000)]
    (tmp_path/'power.jsonl').write_text('\n'.join(canonical_json(row) for row in rows)+'\n')
    fields = ('rail','timestamp_ns','read_start_ns','read_end_ns','current_ma','voltage_mv','power_w','error')
    with (tmp_path/'power.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result['power'].update(raw_sha256=file_sha256(tmp_path/'power.jsonl'), csv_sha256=file_sha256(tmp_path/'power.csv'))
    session = {'status': 'completed', 'exit_code': 0, 'errors': [], 'backend': {'model_instances': 1},
        'collect_power': True, 'device_preflight': {'verified': True, 'observed': {
            'cpu_policies': [{'path': str(i), 'min_khz': 2201600, 'max_khz': 2201600} for i in range(3)],
            'emc_rate_hz': 3199000000, 'emc_cap_hz': 3199000000, 'emc_max_rate_hz': 3199000000,
            'emc_locked': 1, 'emc_halted': 1, 'maxn': {'name': 'MAXN', 'mode_id': 0}, 'tpc': '0',
            'fan_pwm': 255, 'fan_policy': 'user_space', 'fan_service': 'inactive'}}, 'results': [result],
        'counts': {'generation_completed': 1, 'timing_valid': 1, 'energy_valid': 1,
            'generation_failed': 0, 'generation_interrupted': 0, 'not_run': 0, 'energy_not_collected': 0,
            'request_cleanup_failed': 0}, 'provenance': {'runtime': {'source_commit': 'test-source'}}}
    def save():
        write('generation.json', result['generation'])
        write('request.json', result['request'])
        write('result.json', result)
        write('session.json', session)
    save()
    return tmp_path, session, result, save


def test_independent_saved_session_keeps_normal_slo_failure(session_bundle):
    root, session, result, save = session_bundle
    audit = audit_session(root/'session.json')
    assert audit['status'] == 'passed' and audit['slo_violations_preserved'] == 1
    assert audit['raw_power_rows'] == 9 and audit['sensor_response_validated'] is False


@pytest.mark.parametrize('mutation', [None, 'per_rail_rate', 'total_rate', 'tpot', 'forward_count'])
def test_decode_audit_normalizes_every_rail_by_n_minus_one(session_bundle, mutation):
    root, session, result, save = session_bundle
    result['request']['nmax'] = 3
    result['generation'].update(token_ids=[17,23,25], token_ready_ns=[3_000_000_000,3_500_000_000,4_000_000_000],
        n_output_tokens=3, initial_packs=1, decode_forwards=2, final_cache_length=1026)
    result['timing'].update(tN_ns=4_000_000_000, mean_tpot_s=0.5)
    result['energy'].update(total_energy_j=9.0, decode_energy_j=3.0, decode_j_per_token=1.5)
    for rail in RAILS:
        result['energy']['rails'][rail].update(total_energy_j=3.0, decode_energy_j=1.0, decode_j_per_token=0.5)
    predictions = [{'length': 1024, 'group': group, 'frequency_mhz': frequency, 'tpot_s': 0.25,
                    'energy_j_per_token': float(i+1)} for i, (group, frequency) in enumerate(
                    (g,f) for g in (8,16,32,64,128) for f in (408,510,612,714,816,918,1020,1122))]
    result['decisions']['decode'] = {'status': 'active', 'candidates': [{'prediction': row} for row in predictions],
        'selected': predictions[0], 't1_ns': 3_000_000_000, 'td_ns': 3_010_000_000,
        'elapsed_ns': 10_000_000, 'target_s': 2.0, 'budget_s': 1.995, 'selection_path': 'energy_min',
        'quality': [{'group': group, 'score': 0.8} for group in (8,16,32,64,128)],
        'knee': {'status': 'not_found', 'groups': []}}
    result['device']['decode'] = {'verified': True, 'start_ns': 3_020_000_000, 'end_ns': 3_030_000_000,
        'observed': {'min_hz': 408000000, 'max_hz': 408000000, 'cur_hz': 408000000}}
    rows = [dict(rail=rail, timestamp_ns=t, read_start_ns=t, read_end_ns=t, current_ma=1000,
                 voltage_mv=1000, power_w=1.0, error=None) for rail in RAILS for t in (0,2_000_000_000,4_000_000_000,6_000_000_000)]
    (root/'power.jsonl').write_text('\n'.join(canonical_json(row) for row in rows)+'\n')
    with (root/'power.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('rail','timestamp_ns','read_start_ns','read_end_ns','current_ma','voltage_mv','power_w','error'))
        writer.writeheader()
        writer.writerows(rows)
    result['power'].update(raw_sha256=file_sha256(root/'power.jsonl'), csv_sha256=file_sha256(root/'power.csv'))
    if mutation == 'per_rail_rate': result['energy']['rails'][RAILS[0]]['decode_j_per_token'] = 1.0
    elif mutation == 'total_rate': result['energy']['decode_j_per_token'] = 1.0
    elif mutation == 'tpot': result['timing']['mean_tpot_s'] = 1/3
    elif mutation == 'forward_count': result['generation']['decode_forwards'] = 3
    save()
    if mutation is None:
        assert audit_session(root/'session.json')['status'] == 'passed'
    else:
        with pytest.raises(ValueError):
            audit_session(root/'session.json')


@pytest.mark.parametrize('mutation', ['tokens_hash', 'length', 'bool_counter', 'time', 'ttft',
    'n1_decode_energy', 'continued', 'cache_length', 'cleanup', 'pc_select', 'pc_budget',
    'candidate_frequency', 'apply_readback', 'restore_governor', 'energy', 'raw_hash', 'raw_bytes',
    'csv_values', 'input_bytes', 'saved_generation', 'aggregate', 'skipped_count', 'saved_symlink', 'prepared_cpu', 'n1_rail_decode'])
def test_corrupt_saved_session_rejected(session_bundle, mutation):
    root, session, result, save = session_bundle
    if mutation == 'tokens_hash': result['request']['token_ids'][0] = 8
    elif mutation == 'length': result['request']['length'] = 1025
    elif mutation == 'bool_counter': result['generation']['initial_packs'] = False
    elif mutation == 'time': result['generation']['token_ready_ns'][0] = True
    elif mutation == 'ttft': result['timing']['ttft_s'] = 1.99
    elif mutation == 'n1_decode_energy': result['energy']['decode_energy_j'] = 0
    elif mutation == 'continued': result['generation']['stop_reason'] = 'eos'
    elif mutation == 'cache_length': result['generation']['final_cache_length'] = 1025
    elif mutation == 'cleanup': result['cleanup']['cache_released'] = False
    elif mutation == 'pc_select': result['decisions']['prefill']['selected'] = result['decisions']['prefill']['candidates'][1]['prediction']
    elif mutation == 'pc_budget': result['decisions']['prefill']['budget_s'] = 4.0
    elif mutation == 'candidate_frequency': result['decisions']['prefill']['candidates'][-1]['prediction']['frequency_mhz'] = 999
    elif mutation == 'apply_readback': result['device']['prefill']['observed']['cur_hz'] = 510000000
    elif mutation == 'restore_governor': result['device']['restore']['observed']['governor'] = 'userspace'
    elif mutation == 'energy': result['energy']['total_energy_j'] = 6.01
    elif mutation == 'raw_hash': result['power']['raw_sha256'] = 'a'*64
    elif mutation == 'aggregate': session['counts']['generation_completed'] = 2
    elif mutation == 'skipped_count': session['counts']['not_run'] = 1
    elif mutation == 'prepared_cpu': session['device_preflight']['observed']['cpu_policies'][0]['min_khz'] = 1
    elif mutation == 'n1_rail_decode': result['energy']['rails'][RAILS[0]]['decode_energy_j'] = 0
    save()
    if mutation == 'raw_bytes': (root/'power.jsonl').write_text('{}\n')
    elif mutation == 'csv_values':
        (root/'power.csv').write_text((root/'power.csv').read_text().replace('1000', '1001'))
        result['power']['csv_sha256'] = file_sha256(root/'power.csv')
        save()
    elif mutation == 'input_bytes': (root/'input.txt').write_text('changed input')
    elif mutation == 'saved_generation': (root/'generation.json').write_text('{}')
    elif mutation == 'saved_symlink':
        (root/'generation.json').rename(root/'actual-generation.json')
        (root/'generation.json').symlink_to('actual-generation.json')
    with pytest.raises(ValueError):
        audit_session(root/'session.json')


def test_dynamic_fan_and_emc_restoration_allows_only_dynamic_values():
    before = {'cpu': [{'minimum': 1, 'maximum': 2, 'governor': 'schedutil'}],
        'mode': {'name': 'MAXN'}, 'tpc': '0', 'emc': {'lock': 0, 'halt': 0, 'rate': 100},
        'fan_service': 'active', 'fan_policy': 'step_wise', 'fan_pwm': 100}
    after = copy.deepcopy(before)
    after['emc']['rate'] = 200
    after['fan_pwm'] = 120
    assert audit_restoration(before, after, external=True)['status'] == 'passed'
    before['emc']['lock'] = after['emc']['lock'] = 1
    with pytest.raises(ValueError, match='locked EMC'):
        audit_restoration(before, after, external=True)
    before['emc']['lock'] = after['emc']['lock'] = 0
    before['fan_service'] = after['fan_service'] = 'inactive'
    with pytest.raises(ValueError, match='PWM'):
        audit_restoration(before, after, external=True)


def warmup_session(*, first_eos=False):
    request = {'request_id': 'original', 'nmax': 512 if first_eos else 2, 'length': 1024,
        'token_ids_sha256': 'inputhash', 'prompt_sha256': 'prompthash', 'metadata': {'input': 'frozen'}}
    cases = []
    for group in (8,16,32,64,128):
        cases.append({'length': 1024, 'group': group, 'nmax': request['nmax'],
            'requests': [{'request_id': 'original', 'nmax': request['nmax']}], 'metadata': request['metadata'],
            'input_token_sha256': 'inputhash', 'prompt_sha256': 'prompthash',
            'token_ids': [128009] if first_eos else [17,23], 'n_output_tokens': 1 if first_eos else 2,
            'status': 'completed', 'cache_released': True, 'stop_reason': 'eos' if first_eos else 'length',
            'prefill_forwards': 1, 'initial_packs': 0 if first_eos else 1, 'decode_forwards': 0 if first_eos else 1,
            'final_cache_length': 1024 if first_eos else 1025, 'started_ns': 2, 'ended_ns': 4,
            'initial_pack_shape': None if first_eos else {'cache_length': 1024, 'key_residual_length': 0, 'value_residual_length': 128},
            'decode_shapes': [] if first_eos else [{'cache_length': 1024, 'key_residual_length': 0,
                                                 'value_residual_length': 128, 'result_cache_length': 1025}]})
    warmup = {'policy': 'formal', 'formal_coverage_qualified': True,
        'coverage_scope': 'actual_inputs_natural_reachable_shapes', 'errors': [], 'interrupted': False,
        'restore': {'verified': True}, 'prepare': {'verified': True},
        'frequency': {'verified': True, 'observed': dict.fromkeys(('min_hz','max_hz','cur_hz'), 1122000000)},
        'started_ns': 1, 'ended_ns': 5, 'cases': cases}
    return {'warmup': warmup, 'results': [{'request': request, 'timing': {'t0_ns': 6}}]}


@pytest.mark.parametrize('first_eos', [False, True])
def test_independent_warmup_coverage_audits_natural_reachable_shapes(first_eos):
    assert audit_warmup(warmup_session(first_eos=first_eos)) == {'status': 'passed', 'cases': 5}


@pytest.mark.parametrize('mutation', ['missing_group', 'duplicate_group', 'wrong_context', 'cap', 'shape',
                                    'fake_pack', 'continued_eos', 'counter', 'release', 'overlap', 'flag', 'skip'])
def test_forged_warmup_ledger_fails_independent_audit(mutation):
    session = warmup_session()
    warmup, case = session['warmup'], session['warmup']['cases'][0]
    if mutation == 'missing_group': warmup['cases'].pop()
    elif mutation == 'duplicate_group': warmup['cases'][1]['group'] = 8
    elif mutation == 'wrong_context': case['prompt_sha256'] = 'different prompt'
    elif mutation == 'cap': case['nmax'] = 512
    elif mutation == 'shape': case['decode_shapes'][0]['key_residual_length'] = 1
    elif mutation == 'fake_pack': case['initial_packs'] = 0
    elif mutation == 'continued_eos': case['token_ids'] = [128009,23]
    elif mutation == 'counter': case['decode_forwards'] = True
    elif mutation == 'release': case['cache_released'] = False
    elif mutation == 'overlap': warmup['ended_ns'] = 7
    elif mutation == 'flag': warmup['formal_coverage_qualified'] = False
    elif mutation == 'skip': case['status'] = 'skipped'
    with pytest.raises(ValueError):
        audit_warmup(session)


@pytest.fixture
def native_bundle():
    binding = {'native_profile': 'agx2', 'backend_versions': {'torch': 'pinned'},
        'backend_sources': {'models/llama_kivi.py': 'a'*64},
        'model_files': {'model.safetensors': 'b'*64, 'tokenizer.json': 'c'*64},
        'native_binaries': {'gemv': {'module': 'kivi_gemv', 'sha256': 'd'*64}}}
    provenance = {'native_profile': 'agx2', 'model_instances': 1, 'raw_prefill_adapter': True,
        'versions': dict(binding['backend_versions']),
        'sources': {key: {'sha256': value} for key, value in binding['backend_sources'].items()},
        'model_files': {key: {'sha256': value} for key, value in binding['model_files'].items()},
        'binaries': {'kivi_gemv': {'sha256': 'd'*64}},
        'effective_overrides': {'torch_dtype': 'float16', 'do_sample': False, 'num_beams': 1,
            'use_cache': True, 'eos_token_ids': [128001,128008,128009]}}
    return provenance, binding


def test_native_exact_audit_matches_all_sources(native_bundle):
    provenance, binding = native_bundle
    audit_native(provenance, binding, raw_prefill=True)


@pytest.mark.parametrize('mutation', ['model', 'tokenizer', 'kernel', 'source', 'version', 'profile', 'sampling', 'adapter', 'instances'])
def test_native_mismatched_execution_cannot_qualify(native_bundle, mutation):
    provenance, binding = native_bundle
    if mutation == 'model': provenance['model_files']['model.safetensors']['sha256'] = 'x'
    elif mutation == 'tokenizer': provenance['model_files']['tokenizer.json']['sha256'] = 'x'
    elif mutation == 'kernel': provenance['binaries']['kivi_gemv']['sha256'] = 'x'
    elif mutation == 'source': provenance['sources']['models/llama_kivi.py']['sha256'] = 'x'
    elif mutation == 'version': provenance['versions']['torch'] = 'x'
    elif mutation == 'profile': provenance['native_profile'] = 'agx1'
    elif mutation == 'sampling': provenance['effective_overrides']['do_sample'] = True
    elif mutation == 'adapter': provenance['raw_prefill_adapter'] = False
    elif mutation == 'instances': provenance['model_instances'] = True
    with pytest.raises(ValueError):
        audit_native(provenance, binding, raw_prefill=True)
