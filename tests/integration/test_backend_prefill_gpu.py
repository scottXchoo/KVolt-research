"""Opt-in B1 correctness qualification; no GPU imports during collection.

The test itself owns the shared hardware lock. Do not wrap pytest in flock.
Candidate and unchanged reference models are loaded sequentially, never together.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('KVOLT_RUN_B1_GPU') != '1', reason='set KVOLT_RUN_B1_GPU=1 on locked AGX1')]
COHORT_SHA256 = 'e9f0c05e0d21e13356f6a64d1176b03bcff613c708b0b0fd72228f711aadc20d'
DEFAULT_SNAPSHOT = '/home/agx3/kvolt/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'


def _cohort():
    from kvolt_runtime.contracts import RequestSpec
    path = Path(os.environ.get('KVOLT_M1_COHORT', 'artifacts/m1/cohort/inputs.json'))
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == COHORT_SHA256
    rows = json.loads(data)['items']
    selected = [next(row for row in rows if row['length'] == length) for length in (1024, 4096, 8192)]
    requests = []
    for row in selected:
        request = RequestSpec(row['id'], tuple(row['input_ids']), row['prompt_sha256'], 1)
        assert request.length == row['length'] and request.token_sha256 == row['input_hash']
        assert hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest() == request.prompt_sha256
        requests.append((request, row['prompt']))
    return requests


def test_b1_raw_prefill_matches_native_first_token_and_reuses_one_model(monkeypatch):
    import fcntl
    from kvolt_runtime.backend.model import InferenceBackend, load_model, model_identity
    from kvolt_runtime.config import RuntimeConfig
    from kvolt_runtime.contracts import CacheFormat

    requests = _cohort()  # validate all input bytes/IDs before GPU work
    report_path = Path(os.environ.get('KVOLT_B1_REPORT', 'artifacts/m1/b1/prefill-report.json'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'kvolt-b1-prefill-v1', 'cohort_sha256': COHORT_SHA256,
              'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'python': sys.version, 'status': 'running', 'requests': []}
    # Existing project lock; fail immediately if another experiment owns it.
    with Path('/run/lock/kvolt-p3-hardware.lock').open('rb') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pytest.fail('AGX1 hardware lock busy; no model or GPU import attempted')
        try:
            snapshot = Path(os.environ.get('KVOLT_MODEL_SNAPSHOT', DEFAULT_SNAPSHOT))
            loads = []
            def loader(path, config):
                loads.append(str(path))
                return load_model(path, config)
            backend = InferenceBackend(loader=loader)
            backend.load_session(snapshot, RuntimeConfig())
            torch = backend.torch
            source = sys.modules[backend.model.__class__.__module__]
            before = model_identity(backend.model)
            report['candidate_provenance'] = backend.provenance
            quant_calls = []
            def forbidden_quant(*args, **kwargs):
                quant_calls.append(True)
                raise AssertionError('raw Prefill invoked initial quantization')
            positions = []
            def inspect_positions(module, args, kwargs):
                position = kwargs['position_ids']
                positions.append((int(position[0, 0].item()), int(position[0, -1].item())))
                assert kwargs['past_key_value'] is None
            hook = backend.model.model.layers[0].self_attn.register_forward_pre_hook(
                inspect_positions, with_kwargs=True)
            candidate_tokens = []
            # Repeat the first input after the longest one to expose leaked KV/positions.
            sequence = requests + [requests[0]]
            with monkeypatch.context() as patch:
                patch.setattr(source, 'triton_quantize_and_pack_along_last_dim', forbidden_quant)
                for request, prompt in sequence:
                    assert backend.tokenizer(prompt, add_special_tokens=False)['input_ids'] == list(request.token_ids)
                    token = backend.prefill_raw(request)
                    raw = token.cache.require(CacheFormat.RAW)
                    assert token.cache.metadata.length == request.length
                    assert token.cache.metadata.group_size is None
                    assert len(raw) == 32
                    assert all(tuple(t.shape) == (1, 8, request.length, 128)
                               and t.dtype == torch.float16 and str(t.device) == 'cuda:0'
                               for pair in raw for t in pair)
                    assert model_identity(backend.model) == before
                    candidate_tokens.append(token.token_id)
                    report['requests'].append({'id': request.request_id, 'length': request.length,
                        'token_sha256': request.token_sha256, 'candidate_token': token.token_id,
                        'ready_ns': token.ready_ns, 'raw_layers': len(raw), 'raw_dtype': 'float16',
                        'raw_shape': [1, 8, request.length, 128], 'cache_length': token.cache.metadata.length})
                    refs = [weakref.ref(t) for pair in raw for t in pair]
                    del raw
                    backend.release_request(token.cache)
                    assert token.cache.format is CacheFormat.RELEASED
                    gc.collect()
                    assert all(ref() is None for ref in refs), 'released raw KV remains reachable'
            hook.remove()
            assert len(loads) == 1 and quant_calls == []
            assert positions == [(0, request.length - 1) for request, _ in sequence]
            assert candidate_tokens[0] == candidate_tokens[-1]
            assert model_identity(backend.model) == before
            report.update({'candidate_loads': len(loads), 'initial_quant_calls': len(quant_calls),
                           'model_identity_preserved': True, 'positions': positions})
            # No second weight instance while the candidate remains resident.
            candidate_ref = weakref.ref(backend.model)
            del hook, backend
            gc.collect()
            assert candidate_ref() is None, 'candidate model still resident before reference load'
            torch.cuda.empty_cache()
            reference = load_model(snapshot, RuntimeConfig(), raw_prefill=False)
            report['reference_provenance'] = reference.provenance
            reference.model.config.group_size = 32
            for layer in reference.model.model.layers:
                layer.self_attn.group_size = 32
                assert type(layer.self_attn) is source.LlamaFlashAttention_KIVI
            with torch.inference_mode():
                for index, (request, _) in enumerate(sequence):
                    input_ids = torch.tensor([request.token_ids], dtype=torch.long, device='cuda:0')
                    output = reference.model(input_ids=input_ids, past_key_values=None,
                                             use_cache=True, return_dict=True)
                    native_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
                    torch.cuda.synchronize()
                    assert len(output.past_key_values) == 32
                    assert all(len(row) == 9 and row[-1] == request.length for row in output.past_key_values)
                    report['requests'][index]['native_token'] = native_token
                    assert native_token == candidate_tokens[index]
                    del output, input_ids
            del reference
            gc.collect()
            torch.cuda.empty_cache()
            report['status'] = 'passed'
        except BaseException as error:
            report.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
        finally:
            report_path.write_text(json.dumps(report, indent=2) + '\n')
