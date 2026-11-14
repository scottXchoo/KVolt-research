"""Read-only, standard-library audit of the frozen AGX3 validation artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

RAILS = ('VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0')
EXPECTED_SERIAL_SHA = 'c9ecb6270ed595be35f29138221737aa9d120d86538219e2208bf5e68e6f8a61'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_check(root, expected_commit):
    """Bind selected working-tree bytes independently to an external Git commit.

    This auditor does not import the production harness or its source checker.
    Only inference sources, packaging metadata, and these validation tools belong
    to the selected source set; experiment outputs elsewhere are not source.
    """
    root = Path(root).resolve()
    require(re.fullmatch(r'[0-9a-f]{40}', expected_commit) is not None, 'expected full commit SHA')

    def git(*arguments):
        result = subprocess.run(['git', '-C', str(root), *arguments], check=True, capture_output=True)
        return result.stdout

    require(git('rev-parse', '--show-toplevel').decode().strip() == str(root), 'source root must be repository root')
    require(git('rev-parse', 'HEAD').decode().strip() == expected_commit, 'source HEAD mismatch')
    selected = ('src', 'pyproject.toml', 'tools/agx3_gsm8k')
    require(not git('status', '--porcelain', '--untracked-files=all', '--', *selected), 'dirty or untracked source')
    # --ignored is necessary: a source file excluded by .gitignore can still
    # shadow imports, and must not disappear from the audit inventory.
    ignored = [Path(name.decode('utf-8')) for name in
               git('ls-files', '--others', '--ignored', '--exclude-standard', '-z', '--', *selected).split(b'\0') if name]
    require(all('__pycache__' in name.parts and name.suffix in ('.pyc', '.pyo') for name in ignored), 'ignored untracked source')
    files = {}
    for entry in git('ls-tree', '-rz', expected_commit, '--', *selected).split(b'\0'):
        if not entry:
            continue
        metadata, raw_name = entry.split(b'\t', 1)
        mode, kind, _ = metadata.split()
        name = raw_name.decode('utf-8')
        require(mode in (b'100644', b'100755') and kind == b'blob', 'unsupported source entry: ' + name)
        path = root / name
        require(path.resolve().is_relative_to(root), 'source path escape')
        require(not any(candidate.is_symlink() for candidate in (path, *path.parents)
                        if candidate != root and root in candidate.parents), 'source symlink')
        require(path.is_file(), 'missing source: ' + name)
        committed = git('show', expected_commit + ':' + name)
        digest = hashlib.sha256(committed).hexdigest()
        require(sha(path) == digest, 'source bytes mismatch: ' + name)
        files[name] = digest
    require(files and 'pyproject.toml' in files, 'empty/incomplete selected source')
    # Git's stat cache can hide a touched file; every selected blob was compared
    # above, and the final status check catches concurrent index/tree changes.
    require(git('rev-parse', 'HEAD').decode().strip() == expected_commit, 'source HEAD changed')
    require(not git('status', '--porcelain', '--untracked-files=all', '--', *selected), 'source changed during audit')
    return dict(source_sha=expected_commit, source_root=str(root), files=files)


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual), label + ': nonfinite')
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), label + ': mismatch')


def integrate(rows, start, end, max_gap):
    """Piecewise-linear signed watts; integrate clipped segments directly."""
    require(type(start) is int and type(end) is int and 0 <= start < end, 'interval')
    require(type(max_gap) is int and max_gap > 0, 'maximum gap')
    require(set(row['rail'] for row in rows) == set(RAILS), 'rail inventory')
    result = {}
    for rail in RAILS:
        points = [row for row in rows if row['rail'] == rail]
        require(len(points) >= 2, rail + ': insufficient samples')
        for row in points:
            require(row.get('error') is None, rail + ': read error')
            require(type(row['timestamp_ns']) is int and row['timestamp_ns'] >= 0, rail + ': timestamp')
            require(type(row['power_w']) in (int, float) and math.isfinite(row['power_w']), rail + ': power')
        require(all(a['timestamp_ns'] < b['timestamp_ns'] for a, b in zip(points, points[1:])), rail + ': order')
        require(points[0]['timestamp_ns'] < start and points[-1]['timestamp_ns'] > end, rail + ': strict coverage')
        energy = 0.0
        gaps = []
        for a, b in zip(points, points[1:]):
            x, y = max(start, a['timestamp_ns']), min(end, b['timestamp_ns'])
            if x >= y:
                continue
            width = b['timestamp_ns'] - a['timestamp_ns']
            require(width <= max_gap, rail + ': sampling gap')
            gaps.append(width)
            slope = (b['power_w'] - a['power_w']) / width
            left = a['power_w'] + slope * (x - a['timestamp_ns'])
            right = a['power_w'] + slope * (y - a['timestamp_ns'])
            energy += (left + right) * 0.5 * ((y - x) / 1e9)
        require(math.isfinite(energy), rail + ': nonfinite integral')
        result[rail] = dict(energy_j=energy, max_gap_ns=max(gaps), samples=len(points))
    return result


def extracted(text):
    match = re.search(r'#### (\-?[0-9\.\,]+)', text)
    return match.group(1).strip().replace(',', '') if match else None


def audit(attempt, source_root, expected_commit):
    base = attempt.parent
    report = read(attempt / 'gpu-validation.json')
    require(report['status'] == 'passed', 'GPU validation did not pass')
    require(report['source_sha'] == expected_commit, 'frozen source commit')
    require(report['device_serial_sha256'] == EXPECTED_SERIAL_SHA, 'AGX3 physical device')
    model = report['model_audit']
    require(all(model[k] is True for k in ('all_fp16', 'all_cuda0', 'all_eval')), 'FP16/CUDA/eval model')
    require(model['layer_count'] == 32 and model['parameter_tensors'] > 0, 'model layers/parameters')
    require(model['attention_classes'] == ['LlamaFlashAttention_KIVI'], 'KIVI attention class')
    require(model['settings'] == dict(k_bits=2, v_bits=2, group_size=32, residual_length=128), 'KIVI configuration')
    manifest = read(base / 'source-manifest.json')
    source = Path(source_root).resolve()
    actual = source_check(source, expected_commit)
    # The audit may run in a separate clone. Preserve the recorded remote path,
    # while binding the commit and every source byte to the explicit local root.
    require(actual['source_sha'] == manifest['source_sha'] and actual['files'] == manifest['files'], 'source manifest Git/content binding')
    require(report['source_before'] == report['source_after'] == manifest, 'source before/after')
    require(report['script_sha256'] == sha(source / 'tools/agx3_gsm8k/verify.py'), 'harness hash')
    require(report['profile_sha256'] == sha(base / 'profile.json'), 'profile hash')
    require(report['controls_before'] == report['controls_after'] and report['controls_before'], 'control restoration')
    require(report['post_lock_available'] is True and report['remaining_children'] == [], 'lock/children cleanup')
    require(report['lock_acquired_before_torch_import'] is True, 'lock ordering')
    require(report['cleanup']['allocated_after'] == report['cleanup']['reserved_after'] == 0, 'CUDA cleanup')
    require(report.get('cleanup_errors') == [], 'cleanup errors')
    environment = read(attempt / 'environment/summary.json')
    require(environment == report['environment_report'], 'environment artifact consistency')
    require(environment['metadata']['profile'] == read(base / 'profile.json'), 'environment profile')
    require(environment['preparation']['status'] == 'succeeded' and environment['preparation']['warmup_completed'] == 1, 'warmup')
    require(environment['workload']['status'] == environment['restoration']['status'] == 'succeeded', 'workload/restore')
    require(environment['environment']['status'] == 'satisfied' and environment['environment']['violation_count'] == 0, 'environment violation')
    require(not environment['monitor']['errors'], 'environment monitor failure')
    unavailable = set()
    for event in lines(attempt / 'environment/events.jsonl'):
        for probe, reading in event.get('probes', {}).items():
            for metric, reason in reading.get('errors', {}).items():
                unavailable.add((probe, metric, str(reason)))
    require(environment['metadata']['profile_sha256'] == report['profile_sha256'], 'environment profile hash')
    for key in ('formal_qualification', 'full_test_benchmark', 'sensor_response_validated', 'wall_energy_claimed', 'thermal_throttle_qualification'):
        require(report[key] is False, 'claim scope: ' + key)
    dataset = source / 'src/experiments/quality/GSM8K/dataset'
    protocol = read(dataset / 'protocol.json')
    train, test = lines(dataset / 'data/train.jsonl'), lines(dataset / 'data/test.jsonl')
    form = protocol['prompt']
    prefix = form['separator'].join([form['instruction']] + [form['example_template'].format(question=train[i]['question'], reference_solution=train[i]['answer']) for i in protocol['few_shot']['indices']])
    predictions = lines(attempt / 'results/predictions.jsonl')
    summary = read(attempt / 'results/summary.json')
    require(summary == report['runner_summary'] and len(predictions) == summary['evaluated'] == 10, 'first10 count')
    energy_rows = lines(attempt / 'energy-results.jsonl')
    require(energy_rows == report['energy'] and len(energy_rows) == 10, 'energy mapping')
    warmup = lines(attempt / 'warmup.jsonl')
    cases = report['cases']
    require(len(warmup) == 1 and cases[0] == warmup[0] and cases[0]['warmup'] is True, 'warmup case')
    require(len(cases) == 11 and all(c['warmup'] is False for c in cases[1:]), 'request case count')
    require(any(c['quantize_calls'] > 0 and c['matmul_calls'] > 0 for c in cases[1:]), 'KIVI measured execution')
    expected_prompts = []
    recomputed, correct = [], 0
    for index, (prediction, case, measured) in enumerate(zip(predictions, cases[1:], energy_rows)):
        raw = test[index]
        identity = protocol['evaluation']['id_template'].format(index=index)
        require(prediction['id'] == case['id'] == measured['id'] == identity, 'first10 identity')
        require(measured['ordinal'] == index and case['index'] == index + 1, 'request ordinal')
        prompt = prefix + form['separator'] + form['target_template'].format(question=raw['question'])
        expected_prompts.append(prompt)
        require(case['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest(), 'prompt hash')
        require(prediction['question'] == raw['question'] and prediction['answer'] == extracted(raw['answer']), 'reference separation')
        require(case['continuation'] == prediction['prediction'], 'continuation mapping')
        answer = extracted(prediction['prediction'])
        score = float(answer == extracted(raw['answer']))
        require(prediction['predicted_answer'] == answer and prediction['score'] == score, 'GSM8K score')
        correct += int(score)
        inputs, outputs = case['input_token_ids'], case['output_token_ids']
        require(all(type(t) is int and t >= 0 for t in inputs + outputs), 'token IDs')
        require(case['full_sequence_token_ids'] == inputs + outputs, 'token prefix/continuation')
        require(case['input_length'] == len(inputs) and case['output_length'] == len(outputs), 'token lengths')
        require(case['input_ids_sha256'] == hashlib.sha256(json.dumps(inputs, separators=(',', ':')).encode()).hexdigest(), 'input token hash')
        require(1 <= len(outputs) <= 512, 'output bound')
        eos = case['eos_token_ids']
        require(not any(t in eos for t in outputs[:-1]), 'tokens after EOS')
        require((outputs[-1] in eos and case['stop'] == 'eos') or (len(outputs) == 512 and case['stop'] == 'max_new_tokens'), 'stop reason')
        require(case['flags'] == dict(max_new_tokens=512, do_sample=False, num_beams=1, use_cache=True), 'generation flags')
        require(case['inference_mode'] is True and case['supplied_past_key_values'] is False, 'generation cache mode')
        require(case['num_return_sequences'] == 1 and case['return_dict_in_generate'] is False, 'generation output mode')
        require(case['quantize_calls'] >= 0 and case['matmul_calls'] >= 0, 'KIVI counters')
        if len(outputs) > 1:
            require(case['quantize_calls'] > 0 and case['matmul_calls'] > 0, 'KIVI execution')
        relative = measured['energy_artifacts_dir']
        directory = (attempt / relative).resolve()
        require(directory.is_relative_to(attempt.resolve()), 'energy path traversal')
        energy = read(directory / 'energy.json')
        require(energy == measured['energy'] and energy['valid'] is True and energy['completed'] is True, 'energy validity')
        trace = lines(directory / 'power-trace.jsonl')
        with (directory / 'power-samples.csv').open(newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        require(len(trace) == len(csv_rows), 'CSV/JSONL count')
        for row, saved in zip(trace, csv_rows):
            for key in ('rail', 'timestamp_ns', 'read_start_ns', 'read_end_ns', 'current_ma', 'voltage_mv', 'power_w'):
                require(saved[key] == str(row[key]), 'CSV/JSONL ' + key)
            require(row['error'] is None and saved['error'] == '', 'sensor error')
            require(row['read_end_ns'] >= row['read_start_ns'] >= 0 and row['timestamp_ns'] == (row['read_start_ns'] + row['read_end_ns']) // 2, 'sensor read window')
            close(row['power_w'], row['current_ma'] * row['voltage_mv'] / 1e6, 'signed sensor watts')
        values = integrate(trace, energy['start_ns'], energy['end_ns'], report['power_profile']['max_gap_ns'])
        total = sum(v['energy_j'] for v in values.values())
        duration = (energy['end_ns'] - energy['start_ns']) / 1e9
        close(energy['duration_s'], duration, 'duration')
        close(energy['energy_j'], total, 'total joules')
        close(energy['average_power_w'], total / duration, 'average watts')
        for rail, value in values.items():
            close(energy['rails'][rail]['energy_j'], value['energy_j'], rail + ' joules')
            close(energy['rails'][rail]['average_power_w'], value['energy_j'] / duration, rail + ' watts')
        recomputed.append(dict(id=identity, energy_j=total, duration_s=duration, rails=values))
    require(summary['correct'] == correct, 'correct sum')
    close(summary['accuracy'], correct / 10, 'subset accuracy')
    require(summary['first_id'] == predictions[0]['id'] and summary['last_id'] == predictions[-1]['id'], 'summary identity')
    require(summary['dataset']['selected_prompt_sha256'] == hashlib.sha256(''.join(p + '\n' for p in expected_prompts).encode()).hexdigest(), 'selected prompt hash')
    return dict(status='passed', source_sha=expected_commit, source_root=str(source),
                recorded_source_root=manifest.get('source_root'), requests=10,
                correct=correct, subset_accuracy=correct / 10, recomputed_energy=recomputed,
                environment_observations=dict(read_errors=environment['monitor']['read_errors'], samples=environment['monitor']['samples'],
                    unavailable_sources=[dict(probe=p, metric=m, reason=r) for p, m, r in sorted(unavailable)]),
                limitations=['Token IDs and decoded text association use saved evidence; tokenizer execution is not repeated.', 'Sampled INA3221 energy does not establish wall energy or sensor response qualification.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.attempt.resolve(), args.source_root, args.expected_commit)
    except Exception as error:
        result = dict(status='failed', error=dict(type=type(error).__name__, message=str(error)))
    result.update(schema='kvolt.agx3-independent-audit.v1', auditor_sha256=sha(__file__))
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'recomputed_energy'}, ensure_ascii=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
