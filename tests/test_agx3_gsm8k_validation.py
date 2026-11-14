"""CPU-only validation of Git provenance and independent AGX3 artifact audits."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools/agx3_gsm8k'
S = 10**9


def load_module(name):
    spec = importlib.util.spec_from_file_location('agx3_' + name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auditor = load_module('independent_audit')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    (root / 'pyproject.toml').write_text('[project]\nname = "synthetic-fixture"\n')
    dataset = root / 'src/experiments/quality/GSM8K/dataset'
    protocol = dict(prompt=dict(instruction='Solve.', separator='\n\n', example_template='Q: {question}\nA: {reference_solution}', target_template='Q: {question}\nAnswer:'), few_shot=dict(indices=[0]), evaluation=dict(id_template='gsm8k/main/test/{index:04d}'))
    dump(dataset / 'protocol.json', protocol)
    jsonl(dataset / 'data/train.jsonl', [dict(question='Training only', answer='#### 0')])
    jsonl(dataset / 'data/test.jsonl', [dict(question=f'Question {i}', answer=f'#### {i}') for i in range(10)])
    tool = root / 'tools/agx3_gsm8k/verify.py'
    tool.parent.mkdir(parents=True)
    tool.write_text('# synthetic harness bytes\n')
    git(root, 'add', 'src', 'pyproject.toml', 'tools')
    git(root, '-c', 'user.name=CPU Test', '-c', 'user.email=cpu@example.invalid', 'commit', '-qm', 'Synthetic fixture')
    return root, git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def trace():
    rows = []
    for index, rail in enumerate(auditor.RAILS):
        for timestamp in (index * 10_000_000, 2*S + index * 10_000_000, 4*S + index * 10_000_000):
            watts = -2 + timestamp/S if index == 0 else 2 + timestamp/S
            rows.append(dict(rail=rail, timestamp_ns=timestamp, read_start_ns=timestamp, read_end_ns=timestamp,
                             current_ma=watts*1000, voltage_mv=1000, power_w=watts, error=None))
    return rows


def test_signed_ramp_has_independent_rail_timestamps(trace):
    result = auditor.integrate(trace, S, 3*S, 2*S)
    assert result[auditor.RAILS[0]]['energy_j'] == pytest.approx(0)
    assert result[auditor.RAILS[1]]['energy_j'] == pytest.approx(8)
    assert result[auditor.RAILS[2]]['energy_j'] == pytest.approx(8)
    negative = [dict(row, power_w=-1) for row in trace]
    assert sum(value['energy_j'] for value in auditor.integrate(negative, S, 3*S, 2*S).values()) == -6


@pytest.mark.parametrize('change', ['coverage', 'gap', 'readerror', 'duplicate', 'nonfinite'])
def test_integrator_rejects_invalid_trace(trace, change):
    start, gap = S, 2*S
    if change == 'coverage':
        start = 0
    elif change == 'gap':
        gap -= 1
    elif change == 'readerror':
        trace[0]['error'] = {'type': 'OSError'}
    elif change == 'duplicate':
        trace[1]['timestamp_ns'] = trace[0]['timestamp_ns']
    else:
        trace[0]['power_w'] = float('nan')
    with pytest.raises(ValueError):
        auditor.integrate(trace, start, 3*S, gap)


@pytest.mark.parametrize('module_name', ['independent_audit', 'verify'])
def test_source_check_binds_exact_commit(repository, module_name):
    root, commit = repository
    module = load_module(module_name)
    result = module.source_check(root, commit)
    assert result['source_sha'] == commit
    assert result['source_root'] == str(root.resolve())
    assert result['files']['pyproject.toml'] == hashlib.sha256((root / 'pyproject.toml').read_bytes()).hexdigest()
    assert 'tools/agx3_gsm8k/verify.py' in result['files']


@pytest.mark.parametrize('module_name', ['independent_audit', 'verify'])
@pytest.mark.parametrize('change', ['wrong_commit', 'abbreviated_commit', 'dirty', 'staged', 'untracked', 'ignored', 'symlink', 'assume_unchanged'])
def test_source_check_rejects_unbound_sources(repository, module_name, change):
    root, commit = repository
    module = load_module(module_name)
    if change == 'wrong_commit':
        commit = '0' * 40
    elif change == 'abbreviated_commit':
        commit = commit[:8]
    elif change in ('dirty', 'staged', 'assume_unchanged'):
        if change == 'assume_unchanged':
            git(root, 'update-index', '--assume-unchanged', 'pyproject.toml')
        (root / 'pyproject.toml').write_text('changed bytes\n')
        if change == 'staged':
            git(root, 'add', 'pyproject.toml')
    elif change in ('untracked', 'ignored'):
        if change == 'ignored':
            (root / '.gitignore').write_text('src/shadow.py\n')
        (root / 'src/shadow.py').write_text('shadow = True\n')
    else:
        original = root / 'pyproject.toml'
        target = root.parent / 'same-bytes.toml'
        target.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(target)
    with pytest.raises((ValueError, AssertionError)):
        module.source_check(root, commit)


@pytest.mark.parametrize('module_name', ['independent_audit', 'verify'])
def test_source_check_allows_only_ignored_bytecode_cache(repository, module_name):
    root, commit = repository
    module = load_module(module_name)
    (root / '.gitignore').write_text('__pycache__/\n')
    cache = root / 'src/__pycache__'
    cache.mkdir()
    (cache / 'module.cpython-313.pyc').write_bytes(b'synthetic cache')
    assert module.source_check(root, commit)['source_sha'] == commit
    (cache / 'shadow.py').write_text('not bytecode\n')
    with pytest.raises((ValueError, AssertionError)):
        module.source_check(root, commit)


@pytest.mark.parametrize('module_name', ['independent_audit', 'verify'])
def test_import_and_help_have_no_gpu_or_execution(tmp_path, module_name):
    path = TOOLS / (module_name + '.py')
    code = """import importlib.util, runpy, sys
path = sys.argv[1]
before = list(sys.path)
spec = importlib.util.spec_from_file_location('unexecuted', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runpy.run_path(path, run_name='__mp_main__')
assert before == sys.path
assert not ({'torch', 'transformers', 'triton', 'flash_attn'} & set(sys.modules))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', '-c', code, str(path)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-B', str(path), '--help'], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


@pytest.fixture
def attempt(repository, tmp_path, trace):
    root, commit = repository
    base = tmp_path / 'results'
    out = base / 'attempt1'
    out.mkdir(parents=True)
    source = auditor.source_check(root, commit)
    source['source_root'] = '/remote/agx3/clean-clone'
    dump(base / 'source-manifest.json', source)
    profile = dict(schema='experiment-control.profile.v1', controls=[], probes=[], rules=[])
    dump(base / 'profile.json', profile)
    environment = dict(metadata=dict(profile=profile, profile_sha256=auditor.sha(base / 'profile.json')),
                       preparation=dict(status='succeeded', warmup_completed=1), workload=dict(status='succeeded'),
                       restoration=dict(status='succeeded'), environment=dict(status='satisfied', violation_count=0),
                       monitor=dict(errors=[], read_errors=3, samples=1))
    dump(out / 'environment/summary.json', environment)
    jsonl(out / 'environment/events.jsonl', [dict(kind='sample', probes=dict(linux=dict(errors={'throttle': 'source_not_configured', 'gpu.utilization_pct': 'source_not_configured', 'emc.utilization_pct': 'source_not_configured'})))])
    predictions, cases, energies, prompts = [], [], [], []
    values = auditor.integrate(trace, S, 3*S, 2*S)
    for index in range(10):
        identity = f'gsm8k/main/test/{index:04d}'
        text = f'#### {index}'
        prompt = f'Solve.\n\nQ: Training only\nA: #### 0\n\nQ: Question {index}\nAnswer:'
        prompts.append(prompt)
        predictions.append(dict(id=identity, question=f'Question {index}', answer=str(index), prediction=text, predicted_answer=str(index), score=1.0))
        inputs, outputs = [1, 2, 3], ([4, 99] if index else [99])
        cases.append(dict(index=index+1, id=identity, warmup=False, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          continuation=text, input_token_ids=inputs, output_token_ids=outputs, full_sequence_token_ids=inputs+outputs,
                          input_length=3, output_length=len(outputs), input_ids_sha256=hashlib.sha256(json.dumps(inputs, separators=(',', ':')).encode()).hexdigest(),
                          eos_token_ids=[99], stop='eos', flags=dict(max_new_tokens=512, do_sample=False, num_beams=1, use_cache=True),
                          inference_mode=True, supplied_past_key_values=False, num_return_sequences=1, return_dict_in_generate=False,
                          quantize_calls=1 if index else 0, matmul_calls=1 if index else 0))
        rails = {rail: dict(energy_j=value['energy_j'], average_power_w=value['energy_j']/2) for rail, value in values.items()}
        energy = dict(valid=True, completed=True, start_ns=S, end_ns=3*S, duration_s=2, energy_j=16, average_power_w=8, rails=rails)
        relative = f'energy/{index:06d}'
        directory = out / relative
        dump(directory / 'energy.json', energy)
        jsonl(directory / 'power-trace.jsonl', trace)
        with (directory / 'power-samples.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        energies.append(dict(id=identity, ordinal=index, energy=energy, energy_artifacts_dir=relative))
    warmup = copy.deepcopy(cases[0])
    warmup.update(index=0, warmup=True)
    jsonl(out / 'warmup.jsonl', [warmup])
    jsonl(out / 'results/predictions.jsonl', predictions)
    jsonl(out / 'energy-results.jsonl', energies)
    summary = dict(evaluated=10, correct=10, accuracy=1.0, first_id=predictions[0]['id'], last_id=predictions[-1]['id'],
                   dataset=dict(selected_prompt_sha256=hashlib.sha256(''.join(p+'\n' for p in prompts).encode()).hexdigest()))
    dump(out / 'results/summary.json', summary)
    report = dict(status='passed', device_serial_sha256=auditor.EXPECTED_SERIAL_SHA,
                  model_audit=dict(all_fp16=True, all_cuda0=True, all_eval=True, layer_count=32, parameter_tensors=3,
                                   attention_classes=['LlamaFlashAttention_KIVI'], settings=dict(k_bits=2, v_bits=2, group_size=32, residual_length=128)),
                  source_sha=commit, source_before=source, source_after=source,
                  script_sha256=auditor.sha(root / 'tools/agx3_gsm8k/verify.py'), profile_sha256=auditor.sha(base / 'profile.json'),
                  controls_before={'synthetic': 'same'}, controls_after={'synthetic': 'same'}, post_lock_available=True,
                  remaining_children=[], lock_acquired_before_torch_import=True, cleanup=dict(allocated_after=0, reserved_after=0),
                  cleanup_errors=[], environment_report=environment, runner_summary=summary, energy=energies,
                  cases=[warmup]+cases, power_profile=dict(max_gap_ns=2*S))
    for flag in ('formal_qualification', 'full_test_benchmark', 'sensor_response_validated', 'wall_energy_claimed', 'thermal_throttle_qualification'):
        report[flag] = False
    dump(out / 'gpu-validation.json', report)
    return out, root, commit


def test_complete_synthetic_attempt_and_expected_unavailable_sources(attempt):
    out, root, commit = attempt
    result = auditor.audit(out, root, commit)
    assert result['status'] == 'passed' and result['requests'] == 10
    assert result['environment_observations']['read_errors'] == 3
    assert len(result['environment_observations']['unavailable_sources']) == 3


@pytest.mark.parametrize('change', ['csv', 'device', 'model', 'prefix', 'eos', 'control', 'score', 'mapping', 'profile_hash', 'source_hash', 'energy_sum'])
def test_audit_rejects_artifact_tampering(attempt, change):
    out, root, commit = attempt
    report_path = out / 'gpu-validation.json'
    report = auditor.read(report_path)
    if change == 'csv':
        path = out / 'energy/000000/power-samples.csv'
        path.write_text(path.read_text().replace('VDD_GPU_SOC', 'ALTERED', 1))
    elif change == 'device':
        report['device_serial_sha256'] = 'wrong'
    elif change == 'model':
        report['model_audit']['all_fp16'] = False
    elif change == 'prefix':
        report['cases'][2]['full_sequence_token_ids'][0] = 42
    elif change == 'eos':
        report['cases'][2]['output_token_ids'][0] = 99
        report['cases'][2]['full_sequence_token_ids'][-2] = 99
    elif change == 'control':
        report['controls_after']['synthetic'] = 'changed'
    elif change in ('score', 'mapping'):
        path = out / 'results/predictions.jsonl'
        rows = auditor.lines(path)
        rows[0]['score' if change == 'score' else 'id'] = 0.0 if change == 'score' else 'wrong'
        jsonl(path, rows)
    elif change == 'profile_hash':
        (out.parent / 'profile.json').write_text('{}')
    elif change == 'source_hash':
        manifest = auditor.read(out.parent / 'source-manifest.json')
        manifest['files']['pyproject.toml'] = 'wrong'
        dump(out.parent / 'source-manifest.json', manifest)
    else:
        energy = report['energy'][0]['energy']
        energy['energy_j'] += 1
        dump(out / 'energy/000000/energy.json', energy)
        jsonl(out / 'energy-results.jsonl', report['energy'])
    dump(report_path, report)
    with pytest.raises(ValueError):
        auditor.audit(out, root, commit)


def test_score_keeps_first_marker_and_exact_numeric_string():
    assert auditor.extracted('#### 1,000 then #### 2') == '1000'
    assert auditor.extracted('18') is None
    assert auditor.extracted('#### 18.0') != auditor.extracted('#### 18')


@pytest.mark.parametrize('limit', ['1', '9', '11', '1319'])
def test_verify_rejects_other_limits_before_creating_output(tmp_path, limit):
    verify = load_module('verify')
    with pytest.raises(SystemExit) as stopped:
        verify.main(['--expected-commit', '0'*40, '--output-dir', str(tmp_path/'run'), '--limit', limit])
    assert stopped.value.code == 2
    assert not (tmp_path/'run').exists()


def test_cli_writes_new_output_and_refuses_overwrite(attempt, tmp_path):
    out, root, commit = attempt
    output = tmp_path / 'audit.json'
    arguments = ['--attempt', str(out), '--source-root', str(root), '--expected-commit', commit, '--output', str(output)]
    assert auditor.main(arguments) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        auditor.main(arguments)
    assert output.read_bytes() == original
    assert auditor.main(['--attempt', str(out), '--source-root', str(root), '--expected-commit', '0'*40, '--output', str(tmp_path/'failed.json')]) == 1
    assert auditor.read(tmp_path/'failed.json')['status'] == 'failed'
