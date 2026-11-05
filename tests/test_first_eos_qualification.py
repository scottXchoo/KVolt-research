"""CPU-only qualification guards; synthetic tokens do not qualify GPU EOS."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvolt_runtime.contracts import CacheFormat, CacheHandle, CacheMetadata, TokenResult
from kvolt_runtime.qualification.reference import GenerationRecord, generate_candidate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qualify_first_eos', ROOT / 'tools/qualify_first_eos.py')
q = importlib.util.module_from_spec(spec)
# The implementation is deliberately imported without torch/transformers.
spec.loader.exec_module(q)


def frozen(tmp_path, monkeypatch, native_profile='agx1'):
    snapshot = tmp_path / q.SNAPSHOT
    snapshot.mkdir()
    for name in q.TOKENIZER_HASHES:
        (snapshot / name).write_text(name)
    hashes = {name: q.sha256(snapshot / name) for name in q.TOKENIZER_HASHES}
    monkeypatch.setattr(q, 'TOKENIZER_HASHES', hashes)
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: SimpleNamespace(
        ids=[128000] + [ord(c) for c in text[17:]]))
    monkeypatch.setattr(q, 'load_tokenizer', lambda _: (tokenizer, 'test-tokenizers'))
    return q.prepare(snapshot, tmp_path / 'frozen', native_profile=native_profile)


@pytest.mark.parametrize('profile', ['agx1', 'agx2'])
def test_frozen_profile_is_bound_to_policy_plan_and_run_before_gpu(tmp_path, monkeypatch, profile):
    original = copy.deepcopy(q.POLICY)
    inputs = frozen(tmp_path, monkeypatch, profile)
    data, plan = q.load_frozen(inputs, q.sha256(inputs), tmp_path / q.SNAPSHOT, native_profile=profile)
    assert data['policy']['native_profile'] == data['provenance']['native_profile'] == profile
    assert plan['policy']['native_profile'] == profile
    assert q.POLICY == original
    other = 'agx2' if profile == 'agx1' else 'agx1'
    with pytest.raises(RuntimeError, match='policy|profile'):
        q.load_frozen(inputs, q.sha256(inputs), tmp_path / q.SNAPSHOT, native_profile=other)
    output = tmp_path / 'mismatched-run'
    assert q.main(['--snapshot', str(tmp_path / q.SNAPSHOT), '--inputs', str(inputs),
                   '--expected-input-sha256', q.sha256(inputs), '--output', str(output),
                   '--native-profile', other, '--lock', str(tmp_path / 'unused.lock')]) == 1
    assert not (tmp_path / 'unused.lock').exists()
    assert not (inputs.parent / 'run-attempt.json').exists()


def test_first_eos_run_uses_same_profile_for_both_model_paths(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from kvolt_runtime.backend import model
    from kvolt_runtime.qualification import reference
    inputs = frozen(tmp_path, monkeypatch, 'agx2')
    data, plan = q.load_frozen(inputs, q.sha256(inputs), tmp_path / q.SNAPSHOT, native_profile='agx2')
    output = tmp_path / 'run'
    output.mkdir()
    args = SimpleNamespace(inputs=inputs, output=output, snapshot=tmp_path / q.SNAPSHOT,
                           lock=tmp_path / 'lock', kivi_root=tmp_path / 'kivi', native_profile='agx2')
    monkeypatch.setattr(q, 'hardware_lock', lambda path: nullcontext())
    monkeypatch.setattr(q, 'device_report', lambda torch: {})
    monkeypatch.setattr(q, 'verify_loaded_tokenizer', lambda *args: {})
    monkeypatch.setattr(reference, 'generate_reference', lambda *args: GenerationRecord((128009,), 'eos', 0, 2))
    calls = []
    torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None))
    def load(path, config, **kwargs):
        calls.append(('reference', kwargs))
        return model.LoadedModel(object(), torch, object(), {})
    class CandidateReached(Exception): pass
    def candidate(**kwargs):
        calls.append(('candidate', kwargs))
        raise CandidateReached
    monkeypatch.setattr(model, 'load_model', load)
    monkeypatch.setattr(model, 'InferenceBackend', candidate)
    with pytest.raises(CandidateReached):
        q.run(args, data, plan)
    assert [kind for kind, kwargs in calls] == ['reference', 'candidate']
    assert all(kwargs['native_profile'] == 'agx2' for kind, kwargs in calls)


def test_direct_run_rejects_cross_profile_plan_before_attempt_or_lock(tmp_path, monkeypatch):
    inputs = frozen(tmp_path, monkeypatch, 'agx2')
    data, plan = q.load_frozen(inputs, q.sha256(inputs), tmp_path / q.SNAPSHOT, native_profile='agx2')
    plan = copy.deepcopy(plan)
    plan['policy']['native_profile'] = 'agx1'
    def forbidden(*args):
        raise AssertionError('mismatched plan reached reservation/lock')
    monkeypatch.setattr(q, 'reserve_attempt', forbidden)
    monkeypatch.setattr(q, 'hardware_lock', forbidden)
    args = SimpleNamespace(snapshot=tmp_path / q.SNAPSHOT, native_profile='agx2')
    with pytest.raises(RuntimeError, match='plan native build profile'):
        q.run(args, data, plan)


def test_exact_eight_fixed_raw_dialogues_no_trailing_eot():
    assert len(q.PROMPTS) == len(set(q.PROMPTS)) == 8
    assert q.PROMPTS[0].endswith('assistant<|end_header_id|>\n\n2')
    assert q.PROMPTS[-1].endswith('assistant<|end_header_id|>\n\n2')
    assert '1+1은 얼마야? 숫자만 답해.' in q.PROMPTS[-1]
    assert all(p.startswith('<|begin_of_text|><|start_header_id|>user') for p in q.PROMPTS)
    assert all(p.count('<|begin_of_text|>') == p.count('<|eot_id|>') == 1 for p in q.PROMPTS)
    assert not any(p.endswith('<|eot_id|>') for p in q.PROMPTS)


def test_prepare_freezes_all_ids_hashes_cases_and_bounded_repeats(tmp_path, monkeypatch):
    inputs = frozen(tmp_path, monkeypatch)
    data, plan = q.load_frozen(inputs, q.sha256(inputs), tmp_path / q.SNAPSHOT)
    assert len(data['items']) == 8
    assert len(plan['cases']) == 40
    assert len(plan['native_repeat_cases']) == 8
    assert plan['total_generations'] == 88
    assert {c['group'] for c in plan['cases']} == {8, 16, 32, 64, 128}
    assert {c['nmax'] for c in plan['cases']} == {2}
    assert all(row['input_ids'][0] == 128000 for row in data['items'])
    assert all(row['length'] == len(row['input_ids']) for row in data['items'])
    with pytest.raises(RuntimeError, match='empty'):
        q.prepare(tmp_path / q.SNAPSHOT, inputs.parent)


@pytest.mark.parametrize('mutation', ['prompt', 'ids', 'hash', 'length', 'count', 'group', 'nmax', 'repeat'])
def test_frozen_contract_rejects_edits_even_with_new_file_hash(tmp_path, monkeypatch, mutation):
    inputs = frozen(tmp_path, monkeypatch)
    data = json.loads(inputs.read_text())
    if mutation == 'prompt': data['items'][0]['prompt'] += ' '
    if mutation == 'ids': data['items'][0]['input_ids'][0] = 9
    if mutation == 'hash': data['items'][0]['token_sha256'] = '0' * 64
    if mutation == 'length': data['items'][0]['length'] += 1
    if mutation == 'count': data['items'].pop()
    if mutation == 'group': data['policy']['groups'].pop()
    if mutation == 'nmax': data['policy']['nmax'] = 1
    if mutation == 'repeat': data['policy']['native_repeat_group'] = 16
    q.write_json(inputs, data)
    with pytest.raises(RuntimeError):
        q.load_frozen(inputs, q.sha256(inputs), tmp_path / q.SNAPSHOT)


def test_hash_plan_source_and_tokenizer_changes_rejected(tmp_path, monkeypatch):
    inputs = frozen(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match='SHA256'):
        q.load_frozen(inputs, '0' * 64, tmp_path / q.SNAPSHOT)
    planfile = inputs.with_name('plan.json')
    plan = json.loads(planfile.read_text())
    plan['cases'].pop()
    q.write_json(planfile, plan)
    with pytest.raises(RuntimeError, match='plan'):
        q.load_frozen(inputs, q.sha256(inputs), tmp_path / q.SNAPSHOT)
    runtime = q.runtime_identity()
    monkeypatch.setattr(q, 'runtime_identity', lambda: dict(runtime, runtime_commit='changed'))
    with pytest.raises(RuntimeError, match='runtime'):
        q.assert_frozen_identity(json.loads(inputs.read_text()), tmp_path / q.SNAPSHOT)


class Backend:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self._active_cache = None

    def prefill_raw(self, request):
        cache = CacheHandle(CacheMetadata(request.request_id, request.length, 'cuda:0', CacheFormat.RAW), object())
        self._active_cache = cache
        return TokenResult(next(self.tokens), 1, cache)

    def pack_initial(self, cache, group):
        old = cache.metadata
        cache.commit_packed(CacheMetadata(old.request_id, old.length, 'cuda:0', CacheFormat.PACKED, group), object())

    def decode_one(self, cache, previous):
        old = cache.metadata
        cache.commit_decode(CacheMetadata(old.request_id, old.length + 1, 'cuda:0', CacheFormat.PACKED, old.group_size), object())
        return TokenResult(next(self.tokens), 2, cache)

    def release_request(self, cache):
        cache.release()
        self._active_cache = None


@pytest.mark.parametrize('eos', q.EOS_TOKEN_IDS)
def test_first_eos_spy_proves_public_api_and_release_without_forcing(eos):
    request = q.RequestSpec('eos', (128000, 42), 'a' * 64, 2)
    spy = q.CandidateSpy(Backend([eos]))
    actual = generate_candidate(spy, request, 32)
    proof = q.check_pair(actual, actual, request, spy)
    assert proof['branch'] == 'natural_first_eos'
    assert proof['calls'] == {'prefill_raw': 1, 'pack_initial': 0, 'decode_one': 0, 'release_request': 1}
    assert proof['release_before'] == {'format': 'raw_fp16', 'length': 2}
    assert proof['release_after'] == 'released'
    assert proof['active_cache_cleared'] is True


@pytest.mark.parametrize('tokens,stop', [([42, 43], 'length'), ([42, 128009], 'eos')])
def test_nonfirst_eos_normal_branch_is_never_reported_as_first_eos(tokens, stop):
    request = q.RequestSpec('normal', (128000, 42), 'a' * 64, 2)
    spy = q.CandidateSpy(Backend(tokens))
    actual = generate_candidate(spy, request, 8)
    assert actual == GenerationRecord(tuple(tokens), stop, 1, 3)
    assert q.check_pair(actual, actual, request, spy)['branch'] == 'normal_decode'


def test_pair_mismatch_and_invalid_release_are_technical_errors():
    request = q.RequestSpec('bad', (128000, 42), 'a' * 64, 2)
    spy = q.CandidateSpy(Backend([128009]))
    actual = generate_candidate(spy, request, 8)
    with pytest.raises(RuntimeError, match='mismatch'):
        q.check_pair(GenerationRecord((42,), 'eos', 0, 2), actual, request, spy)
    spy.calls['pack_initial'] = 1
    with pytest.raises(RuntimeError, match='first EOS'):
        q.check_pair(actual, actual, request, spy)


def test_summary_requires_actual_first_eos_in_every_group():
    results = [{'key': str(g), 'group': g, 'branch': 'normal_decode'} for g in q.GROUPS]
    summary = q.summarize(results)
    assert summary['first_eos_gpu_qualified'] is False
    assert summary['outcome'] == 'bounded_search_complete_no_first_eos'
    results[0]['branch'] = 'natural_first_eos'
    assert q.summarize(results)['first_eos_gpu_qualified'] is False
    results = [{'key': str(i), 'group': g, 'branch': 'natural_first_eos'} for i, g in enumerate(q.GROUPS) for _ in range(8)]
    assert q.summarize(results)['first_eos_gpu_qualified'] is True


def test_attempt_reservation_has_no_retry(tmp_path):
    q.reserve_attempt(tmp_path, tmp_path / 'results')
    with pytest.raises(RuntimeError, match='attempt'):
        q.reserve_attempt(tmp_path, tmp_path / 'different-results')


def test_busy_lock_records_error_without_gpu_import_or_second_attempt(tmp_path, monkeypatch):
    import fcntl
    import importlib.abc
    import sys

    inputs = frozen(tmp_path, monkeypatch)
    lockpath = tmp_path / 'hardware.lock'
    output = tmp_path / 'gpu-output'
    blocked = []

    class RejectGPU(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('torch', 'transformers', 'triton') or fullname == 'kvolt_runtime.backend.model':
                blocked.append(fullname)
                raise AssertionError('GPU/model import before lock: ' + fullname)

    monkeypatch.setattr(sys, 'meta_path', [RejectGPU()] + sys.meta_path)
    with lockpath.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = q.main(['--snapshot', str(tmp_path / q.SNAPSHOT), '--inputs', str(inputs),
                       '--expected-input-sha256', q.sha256(inputs), '--output', str(output), '--lock', str(lockpath)])
    assert code == 1 and blocked == []
    error = json.loads((output / 'error.json').read_text())
    assert error['outcome'] == 'technical_error'
    assert 'hardware lock busy' in error['traceback']
    assert error['first_eos_gpu_qualified'] is False
    assert not (output / 'summary.json').exists()
    with pytest.raises(RuntimeError, match='attempt'):
        q.reserve_attempt(inputs.parent, tmp_path / 'different-output')


def test_prepare_in_fresh_process_blocks_gpu_and_transformers_imports(tmp_path):
    import subprocess
    import sys

    script = r'''
import importlib.abc, json, pathlib, sys, types
class RejectGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers', 'triton') or fullname == 'kvolt_runtime.backend.model':
            raise AssertionError('CPU preparation imported GPU/model dependency: ' + fullname)
sys.meta_path.insert(0, RejectGPU())
sys.path.insert(0, 'tools')
import qualify_first_eos as q
snapshot = pathlib.Path(sys.argv[1]) / q.SNAPSHOT
snapshot.mkdir()
for name in q.TOKENIZER_HASHES:
    (snapshot / name).write_text(name)
q.TOKENIZER_HASHES = {name: q.sha256(snapshot / name) for name in q.TOKENIZER_HASHES}
class Tokenizer:
    @staticmethod
    def from_file(path):
        assert path.endswith('/tokenizer.json')
        return Tokenizer()
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return types.SimpleNamespace(ids=[128000] + [ord(c) for c in text[17:]])
sys.modules['tokenizers'] = types.SimpleNamespace(Tokenizer=Tokenizer)
q.importlib.metadata.version = lambda _: 'fake-tokenizer-only'
assert q.main(['--prepare-only', '--snapshot', str(snapshot), '--output', str(pathlib.Path(sys.argv[1]) / 'freeze')]) == 0
assert not any(name in sys.modules for name in ('torch', 'transformers', 'triton', 'kvolt_runtime.backend.model'))
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=ROOT, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['GPU_or_model_loaded'] is False


def test_tokenizer_file_mutation_is_detected_after_freeze(tmp_path, monkeypatch):
    inputs = frozen(tmp_path, monkeypatch)
    (tmp_path / q.SNAPSHOT / 'tokenizer.json').write_text('changed')
    with pytest.raises(RuntimeError, match='tokenizer SHA256'):
        q.load_frozen(inputs, q.sha256(inputs), tmp_path / q.SNAPSHOT)


def test_loaded_tokenizer_must_match_every_frozen_id_without_extra_bos(tmp_path, monkeypatch):
    inputs = frozen(tmp_path, monkeypatch)
    data = json.loads(inputs.read_text())
    by_prompt = {row['prompt']: row['input_ids'] for row in data['items']}
    calls = []

    def tokenizer(text, *, add_special_tokens):
        calls.append((text, add_special_tokens))
        return {'input_ids': by_prompt[text]}

    assert q.verify_loaded_tokenizer(tokenizer, data)['all_eight_exact'] is True
    assert len(calls) == 8 and all(special is False for _, special in calls)
    by_prompt[data['items'][-1]['prompt']] = [128000] + data['items'][-1]['input_ids']
    with pytest.raises(RuntimeError, match='loaded tokenizer disagrees'):
        q.verify_loaded_tokenizer(tokenizer, data)
