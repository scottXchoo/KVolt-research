#!/usr/bin/env python3
"""Bounded synthetic natural-first-EOS diagnostic: freeze on CPU, then run once."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from kvolt_runtime.config import EOS_TOKEN_IDS, GROUPS, SNAPSHOT, RuntimeConfig
from kvolt_runtime.contracts import CacheFormat, RequestSpec, token_ids_sha256
from kvolt_runtime.qualification.compare import compare_generation
from qualify_backend import device_report, hardware_lock, write_json

DIALOGUES = (
    ('What is 1 + 1? Answer with the number only.', '2'),
    ('Reply with exactly OK.', 'OK'),
    ('What is the chemical symbol for oxygen? Answer with the symbol only.', 'O'),
    ('Translate cat into Spanish. Give only the translated word.', 'gato'),
    ('Say only hello.', 'hello'),
    ('What is the opposite of hot? Answer with one word.', 'cold'),
    ('Return exactly an empty JSON object.', '{}'),
    ('1+1은 얼마야? 숫자만 답해.', '2'),
)
PROMPTS = tuple('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' + user
                + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' + answer
                for user, answer in DIALOGUES)
TOKENIZER_HASHES = {
    'tokenizer_config.json': '177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424',
    'tokenizer.json': '79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4',
}
POLICY = {'groups': list(GROUPS), 'nmax': 2, 'native_repeat_group': 32,
          'snapshot': SNAPSHOT, 'eos_token_ids': list(EOS_TOKEN_IDS), 'dtype': 'float16',
          'greedy': True, 'add_special_tokens': False, 'independent_prompt_count': 8,
          'synthetic_diagnostic_only': True, 'performance_or_energy_claims': False,
          'model_or_logits_modification': False, 'extra_search_or_retry': False}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity():
    paths = sorted(set(ROOT.glob('src/**/*.py')) | set(ROOT.glob('tools/**/*.py'))
                   | {ROOT / 'tests/test_first_eos_qualification.py', ROOT / 'docs/backend-qualification.md'})
    return {'root': str(ROOT),
            'runtime_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'runtime_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def tokenizer_identity(snapshot):
    snapshot = Path(snapshot).resolve()
    if snapshot.name != SNAPSHOT:
        raise RuntimeError('tokenizer directory must name the approved snapshot')
    files = {}
    for name, expected in TOKENIZER_HASHES.items():
        if sha256(snapshot / name) != expected:
            raise RuntimeError('pinned tokenizer SHA256 mismatch: ' + name)
        files[name] = expected
    return {'snapshot': str(snapshot), 'revision': SNAPSHOT, 'files_sha256': files}


def load_tokenizer(snapshot):
    # Rust tokenizer only: do not import transformers, torch, or model modules.
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(snapshot) / 'tokenizer.json')), importlib.metadata.version('tokenizers')


def encode_prompts(tokenizer):
    rows = []
    for index, prompt in enumerate(PROMPTS, 1):
        ids = tuple(tokenizer.encode(prompt, add_special_tokens=False).ids)
        if not ids or ids[0] != 128000 or ids.count(128000) != 1:
            raise RuntimeError('exactly one existing BOS is required; no extra BOS allowed')
        request = RequestSpec(f'completed-{index:02d}', ids, hashlib.sha256(prompt.encode('utf-8')).hexdigest(), 2)
        if request.length > 8192:
            raise RuntimeError('diagnostic input exceeds backend capacity')
        rows.append({'id': request.request_id, 'prompt': prompt, 'prompt_sha256': request.prompt_sha256,
                     'input_ids': list(ids), 'token_sha256': request.token_sha256, 'length': request.length})
    if len({tuple(row['input_ids']) for row in rows}) != 8:
        raise RuntimeError('diagnostic requires eight distinct token sequences')
    return rows


def make_plan(data, input_sha):
    cases = [{'key': f'{row["id"]}-g{group}-n2', 'request_id': row['id'], 'group': group,
              'nmax': 2, 'length': row['length'], 'prompt_sha256': row['prompt_sha256'],
              'token_sha256': row['token_sha256']} for row in data['items'] for group in GROUPS]
    return {'schema': 'kvolt.m1.first-eos.plan.v1', 'inputs_sha256': input_sha,
            'policy': data['policy'], 'provenance': data['provenance'], 'cases': cases,
            'native_repeat_cases': [c['key'] for c in cases if c['group'] == 32],
            'paired_generations': 80, 'native_repeat_generations': 8, 'total_generations': 88,
            'initial_cache_proof': 'out of scope for non-EOS paths; existing B2 suites cover packing'}


def require_empty(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise RuntimeError('output must be empty; no resume, new attempt, or automatic retry')


def prepare(snapshot, output):
    require_empty(output)
    identity = tokenizer_identity(snapshot)
    tokenizer, version = load_tokenizer(snapshot)
    rows = encode_prompts(tokenizer)
    data = {'schema': 'kvolt.m1.first-eos.inputs.v1', 'policy': POLICY,
            'provenance': {'runtime': runtime_identity(), 'tokenizer': dict(identity, tokenizers_version=version)},
            'items': rows}
    Path(output).mkdir(parents=True, exist_ok=True)
    inputs = Path(output) / 'inputs.json'
    write_json(inputs, data)
    write_json(Path(output) / 'plan.json', make_plan(data, sha256(inputs)))
    assert_frozen_identity(data, snapshot)
    return inputs


def assert_frozen_identity(data, snapshot):
    if data['provenance']['runtime'] != runtime_identity():
        raise RuntimeError('runtime root/commit/status/source changed since CPU freeze')
    expected = data['provenance']['tokenizer']
    identity = tokenizer_identity(snapshot)
    if {k: expected[k] for k in identity} != identity:
        raise RuntimeError('tokenizer snapshot changed since CPU freeze')


def load_frozen(inputs, expected_sha, snapshot):
    if len(expected_sha) != 64 or sha256(inputs) != expected_sha:
        raise RuntimeError('frozen inputs SHA256 mismatch')
    data = json.loads(Path(inputs).read_text(encoding='utf-8'))
    if data.get('schema') != 'kvolt.m1.first-eos.inputs.v1' or data.get('policy') != POLICY:
        raise RuntimeError('frozen diagnostic policy mismatch')
    rows = data.get('items', [])
    if len(rows) != 8 or [row['prompt'] for row in rows] != list(PROMPTS):
        raise RuntimeError('frozen diagnostic requires exactly eight approved prompts in order')
    for index, row in enumerate(rows, 1):
        ids = tuple(row['input_ids'])
        request = RequestSpec(f'completed-{index:02d}', ids, hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest(), 2)
        if (row['id'] != request.request_id or row['prompt_sha256'] != request.prompt_sha256
                or row['token_sha256'] != token_ids_sha256(ids) or row['length'] != request.length
                or request.length > 8192 or ids[0] != 128000 or ids.count(128000) != 1):
            raise RuntimeError('frozen prompt/token ID/length/hash mismatch')
    if len({tuple(row['input_ids']) for row in rows}) != 8:
        raise RuntimeError('frozen token sequences must be distinct')
    assert_frozen_identity(data, snapshot)
    # Re-encoding is CPU-only and proves IDs correspond to the pinned tokenizer.
    tokenizer, version = load_tokenizer(snapshot)
    if rows != encode_prompts(tokenizer) or data['provenance']['tokenizer']['tokenizers_version'] != version:
        raise RuntimeError('frozen tokenizer encoding/version mismatch')
    plan = make_plan(data, expected_sha)
    if json.loads(Path(inputs).with_name('plan.json').read_text(encoding='utf-8')) != plan:
        raise RuntimeError('frozen plan mismatch')
    return data, plan


def reserve_attempt(directory, output):
    try:
        with (Path(directory) / 'run-attempt.json').open('x') as stream:
            json.dump({'output': str(Path(output).resolve()), 'pid': os.getpid(),
                       'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
    except FileExistsError as error:
        raise RuntimeError('frozen plan already has a run attempt; no new attempt or retry') from error


def verify_loaded_tokenizer(tokenizer, data):
    """Confirm the loader's AutoTokenizer agrees without any model forward."""
    hashes = {}
    for row in data['items']:
        ids = tuple(tokenizer(row['prompt'], add_special_tokens=False)['input_ids'])
        if ids != tuple(row['input_ids']):
            raise RuntimeError('loaded tokenizer disagrees with frozen IDs: ' + row['id'])
        hashes[row['id']] = token_ids_sha256(ids)
    return {'all_eight_exact': True, 'add_special_tokens': False, 'token_sha256': hashes}


class CandidateSpy:
    """Delegate public request APIs unchanged; never intercept model or logits."""
    def __init__(self, backend):
        self.backend = backend
        self.calls = dict(prefill_raw=0, pack_initial=0, decode_one=0, release_request=0)
        self.release_before = self.release_after = self.active_cache_cleared = None
        self.observed_token_ids = []

    def prefill_raw(self, request):
        self.calls['prefill_raw'] += 1
        result = self.backend.prefill_raw(request)
        self.observed_token_ids.append(result.token_id)
        return result

    def pack_initial(self, cache, group):
        self.calls['pack_initial'] += 1
        return self.backend.pack_initial(cache, group)

    def decode_one(self, cache, previous):
        self.calls['decode_one'] += 1
        result = self.backend.decode_one(cache, previous)
        self.observed_token_ids.append(result.token_id)
        return result

    def release_request(self, cache):
        self.calls['release_request'] += 1
        self.release_before = {'format': cache.format.value, 'length': cache.metadata.length}
        try:
            return self.backend.release_request(cache)
        finally:
            self.release_after = cache.format.value
            self.active_cache_cleared = self.backend._active_cache is None

    def evidence(self):
        return {'calls': dict(self.calls), 'release_before': self.release_before,
                'release_after': self.release_after, 'active_cache_cleared': self.active_cache_cleared,
                'observed_token_ids': list(self.observed_token_ids)}


def check_pair(expected, actual, request, spy):
    difference = compare_generation(expected, actual)
    if difference:
        raise RuntimeError('generation mismatch: ' + json.dumps(difference))
    if (not actual.token_ids or len(actual.token_ids) > 2 or actual.decode_forwards != len(actual.token_ids) - 1
            or actual.final_cache_length != request.length + actual.decode_forwards):
        raise RuntimeError('generation length/forward/T contract failed')
    first_eos = actual.token_ids[0] in EOS_TOKEN_IDS
    proof = spy.evidence()
    if tuple(proof['observed_token_ids']) != actual.token_ids:
        raise RuntimeError('spy observed token IDs disagree with generation record')
    calls = {'prefill_raw': 1, 'pack_initial': int(not first_eos),
             'decode_one': int(not first_eos), 'release_request': 1}
    if first_eos:
        if (len(actual.token_ids) != 1 or actual.stop_reason != 'eos' or actual.decode_forwards != 0
                or actual.final_cache_length != request.length or proof['calls'] != calls
                or proof['release_before'] != {'format': CacheFormat.RAW.value, 'length': request.length}):
            raise RuntimeError('natural first EOS must skip packing/decode and release RAW T=L')
    elif (len(actual.token_ids) != 2 or proof['calls'] != calls
          or actual.stop_reason != ('eos' if actual.token_ids[-1] in EOS_TOKEN_IDS else 'length')):
        raise RuntimeError('non-first-EOS diagnostic must complete normal Decode at Nmax=2')
    if proof['release_after'] != CacheFormat.RELEASED.value or proof['active_cache_cleared'] is not True:
        raise RuntimeError('release must leave RELEASED cache and clear active backend cache')
    return dict(proof, branch='natural_first_eos' if first_eos else 'normal_decode')


def summarize(results):
    counts = {str(g): {'compared_pairs': 0, 'natural_first_eos': 0, 'normal_decode': 0} for g in GROUPS}
    for row in results:
        count = counts[str(row['group'])]
        count['compared_pairs'] += 1
        count[row['branch']] += 1
    complete = len(results) == 40 and all(c['compared_pairs'] == 8 for c in counts.values())
    first_eos = sum(c['natural_first_eos'] for c in counts.values())
    qualified = complete and all(c['natural_first_eos'] > 0 for c in counts.values())
    return {'outcome': ('natural_first_eos_qualified' if qualified else
                        'bounded_search_complete_no_first_eos' if first_eos == 0 else
                        'bounded_search_complete_partial_first_eos_coverage'),
            'passed_pairs': len(results), 'independent_prompts': 8, 'native_repeats': 8,
            'all_tokens_stop_decode_count_T_exact': complete,
            'first_eos_cases': [r['key'] for r in results if r['branch'] == 'natural_first_eos'],
            'coverage_per_group': counts, 'first_eos_gpu_qualified': qualified,
            'performance_or_energy_claims': False, 'extra_search_or_retry': False}


def run(args, data, plan):
    from kvolt_runtime.qualification.reference import generate_reference, generate_candidate
    cases = [(c, RequestSpec(c['request_id'], tuple(data['items'][i // 5]['input_ids']), c['prompt_sha256'], 2))
             for i, c in enumerate(plan['cases'])]
    inputs_sha, plan_sha = sha256(args.inputs), sha256(args.inputs.with_name('plan.json'))

    def guard():
        if sha256(args.inputs) != inputs_sha or sha256(args.inputs.with_name('plan.json')) != plan_sha:
            raise RuntimeError('frozen input/plan changed during qualification')
        assert_frozen_identity(data, args.snapshot)

    reserve_attempt(args.inputs.parent, args.output)
    with hardware_lock(args.lock):
        guard()  # Before any torch/model import, including on busy-lock failure.
        from kvolt_runtime.backend.model import InferenceBackend, load_model
        write_json(args.output / 'plan.json', plan)
        write_json(args.output / 'frozen-inputs.json', data)
        reference, results, repeats = {}, [], []
        loaded = None
        torch = None
        try:
            guard()
            loaded = load_model(args.snapshot, RuntimeConfig(), kivi_root=args.kivi_root, raw_prefill=False)
            torch = loaded.torch
            write_json(args.output / 'native-provenance.json', loaded.provenance)
            write_json(args.output / 'device.json', device_report(torch))
            write_json(args.output / 'native-tokenizer-agreement.json', verify_loaded_tokenizer(loaded.tokenizer, data))
            for index, (case, request) in enumerate(cases, 1):
                print(f'native {index}/40 {case["key"]}', flush=True)
                record = generate_reference(loaded, request, case['group'])
                reference[case['key']] = record
                write_json(args.output / (case['key'] + '.native.json'), asdict(record))
            for case, request in cases:
                if case['key'] not in plan['native_repeat_cases']:
                    continue
                record = generate_reference(loaded, request, case['group'])
                write_json(args.output / (case['key'] + '.repeat.json'), asdict(record))
                difference = compare_generation(reference[case['key']], record)
                if difference:
                    write_json(args.output / 'first-mismatch.json', {'phase': 'native_repeat', 'case': case['key'], 'difference': difference})
                    raise RuntimeError('native baseline is not repeatable; stop without retry')
                repeats.append(case['key'])
        finally:
            loaded = None
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        backend = None
        try:
            guard()
            backend = InferenceBackend(kivi_root=args.kivi_root)
            backend.load_session(args.snapshot, RuntimeConfig())
            write_json(args.output / 'candidate-provenance.json', backend.provenance)
            write_json(args.output / 'candidate-tokenizer-agreement.json', verify_loaded_tokenizer(backend.tokenizer, data))
            for index, (case, request) in enumerate(cases, 1):
                print(f'candidate {index}/40 {case["key"]}', flush=True)
                spy = CandidateSpy(backend)
                actual = None
                try:
                    actual = generate_candidate(spy, request, case['group'])
                    write_json(args.output / (case['key'] + '.candidate.json'), asdict(actual))
                    proof = check_pair(reference[case['key']], actual, request, spy)
                except Exception:
                    difference = None if actual is None else compare_generation(reference[case['key']], actual)
                    write_json(args.output / 'first-mismatch.json', {'phase': 'candidate', 'case': case['key'],
                               'difference': difference, 'spy': spy.evidence()})
                    raise
                finally:
                    write_json(args.output / (case['key'] + '.spy.json'), spy.evidence())
                # First EOS has no packing; requested G is a comparison label,
                # not evidence that quantization ran in this branch.
                result = dict(proof, key=case['key'], group=case['group'], requested_group=case['group'],
                              quantization_applied=proof['calls']['pack_initial'] == 1,
                              request_id=request.request_id)
                results.append(result)
                write_json(args.output / 'progress.json', {'passed_pairs': len(results), 'total_pairs': 40, 'cases': results})
        finally:
            spy = backend = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        guard()
        summary = summarize(results)
        summary['native_repeat_cases'] = repeats
        summary['source_input_plan_unchanged_at_end'] = True
        write_json(args.output / 'summary.json', summary)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--expected-input-sha256')
    parser.add_argument('--kivi-root', type=Path, default=Path('/home/agx3/kvolt/third_party/KIVI'))
    parser.add_argument('--lock', type=Path, default=Path('/run/lock/kvolt-p3-hardware.lock'))
    args = parser.parse_args(argv)
    require_empty(args.output)
    if args.prepare_only and (args.inputs or args.expected_input_sha256):
        parser.error('--prepare-only does not accept frozen run inputs')
    if not args.prepare_only and (args.inputs is None or args.expected_input_sha256 is None):
        parser.error('GPU run requires --inputs and --expected-input-sha256 from the CPU freeze')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = None
    try:
        if args.prepare_only:
            inputs = prepare(args.snapshot, args.output)
            print(json.dumps({'inputs': str(inputs), 'inputs_sha256': sha256(inputs), 'GPU_or_model_loaded': False}))
            return 0
        data, plan = load_frozen(args.inputs, args.expected_input_sha256, args.snapshot)
        return run(args, data, plan)
    except Exception as error:
        not_run = {}
        if plan is not None:
            for suffix, keys in [('native', [c['key'] for c in plan['cases']]),
                                 ('candidate', [c['key'] for c in plan['cases']]),
                                 ('repeat', plan['native_repeat_cases'])]:
                not_run[suffix] = [key for key in keys if not (args.output / (key + '.' + suffix + '.json')).exists()]
        write_json(args.output / 'error.json', {'outcome': 'technical_error', 'error_type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc(), 'first_eos_gpu_qualified': False,
                   'automatic_retry': False, 'not_completed_cases': not_run})
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
