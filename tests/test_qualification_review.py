"""Independent CPU checks of the actual native wrapper and CLI lock boundary."""
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvolt_runtime.contracts import RequestSpec
from kvolt_runtime.errors import QualificationError
from kvolt_runtime.qualification.compare import compare_generation
from kvolt_runtime.qualification.reference import GenerationRecord, generate_candidate, generate_reference


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def all(self):
        return self


class Logits:
    def __init__(self, token):
        self.token = token

    def __getitem__(self, index):
        assert index == (slice(None), -1, slice(None))
        return self

    def argmax(self, dim):
        assert dim == -1
        return Scalar(self.token)


def native_fixture(tokens, *, corrupt_layer=None, finite=True):
    calls, syncs = [], []
    config = SimpleNamespace(group_size=None)
    attentions = [SimpleNamespace(k_bits=2, v_bits=2, residual_length=128,
                                 config=SimpleNamespace(group_size=None))
                  for _ in range(32)]

    class Model:
        def __call__(self, **options):
            calls.append(options)
            index = len(calls) - 1
            past = options['past_key_values']
            length = len(options['input_ids'][0]) if past is None else past[0][-1] + 1
            cache = tuple((None,) * 8 + (length + int(layer == corrupt_layer),)
                          for layer in range(32))
            return SimpleNamespace(logits=Logits(tokens[index]), past_key_values=cache)

    model = Model()
    model.config = config
    model.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=a) for a in attentions])
    torch = SimpleNamespace(long='long', inference_mode=nullcontext,
        tensor=lambda values, **options: values, isfinite=lambda _: Scalar(finite),
        cuda=SimpleNamespace(synchronize=lambda: syncs.append('sync')))
    loaded = SimpleNamespace(model=model, torch=torch,
                             provenance={'raw_prefill_adapter': False})
    return loaded, calls, syncs, attentions


@pytest.mark.parametrize('cap', [1, 2, 3, 128, 512])
def test_actual_native_wrapper_consumes_y1_at_L_and_stops_at_cap(cap):
    tokens = tuple(range(17, 17 + cap))
    loaded, calls, syncs, attentions = native_fixture(tokens)
    snapshots = []
    request = RequestSpec('native-cap', (1,) * 1024, 'a' * 64, cap)
    record = generate_reference(loaded, request, 8, initial_cache=snapshots.append)
    assert record == GenerationRecord(tokens, 'length', cap - 1, 1024 + cap - 1)
    assert len(calls) == cap
    assert calls[0]['input_ids'] == [request.token_ids]
    assert calls[0]['past_key_values'] is None
    assert 'position_ids' not in calls[0]
    for index, call in enumerate(calls[1:], start=1):
        assert call['input_ids'] == [[tokens[index - 1]]]
        assert call['position_ids'] == [[1024 + index - 1]]
        assert call['past_key_values'][0][-1] == 1024 + index - 1
    assert len(snapshots) == int(cap > 1)
    if snapshots:
        assert snapshots[0][31][-1] == 1024
    assert len(syncs) == cap + 1
    assert loaded.model.config.group_size == 8
    assert all(a.group_size == a.config.group_size == 8 for a in attentions)


@pytest.mark.parametrize('eos', [128001, 128008, 128009])
@pytest.mark.parametrize('prefix', [(), (17,), (17, 19)])
def test_actual_native_wrapper_preserves_all_natural_eos_without_extra_forward(eos, prefix):
    tokens = prefix + (eos,)
    loaded, calls, _, _ = native_fixture(tokens)
    record = generate_reference(loaded, RequestSpec('native-eos', (1,) * 1024, 'b' * 64, 512), 128)
    assert record == GenerationRecord(tokens, 'eos', len(prefix), 1024 + len(prefix))
    assert len(calls) == len(tokens)


def test_actual_native_eos_has_priority_over_cap():
    loaded, calls, _, _ = native_fixture((128001,))
    record = generate_reference(loaded, RequestSpec('native-eos-cap', (1,) * 1024, 'c' * 64, 1), 16)
    assert record.stop_reason == 'eos' and len(calls) == 1


@pytest.mark.parametrize('eos', [128001, 128008, 128009])
def test_candidate_all_first_eos_skip_packing_and_callback(eos):
    events = []
    cache = SimpleNamespace(metadata=SimpleNamespace(length=1024))
    def forbidden(*args):
        pytest.fail('first EOS must not pack, decode, or inspect packed cache')
    backend = SimpleNamespace(prefill_raw=lambda _: SimpleNamespace(cache=cache, token_id=eos),
        pack_initial=forbidden, decode_one=forbidden, release_request=events.append)
    record = generate_candidate(backend, RequestSpec('first-eos', (1,) * 1024, 'f' * 64, 512),
                                8, initial_cache=forbidden)
    assert record == GenerationRecord((eos,), 'eos', 0, 1024)
    assert events == [cache]


def test_candidate_releases_cache_when_strict_initial_comparison_fails():
    events = []
    cache = SimpleNamespace(metadata=SimpleNamespace(length=1024), require=lambda _: 'packed')
    def mismatch(packed):
        assert packed == 'packed'
        raise QualificationError('first byte differs')
    backend = SimpleNamespace(prefill_raw=lambda _: SimpleNamespace(cache=cache, token_id=17),
        pack_initial=lambda *_: events.append('pack'), release_request=lambda _: events.append('release'))
    with pytest.raises(QualificationError, match='first byte differs'):
        generate_candidate(backend, RequestSpec('cache-mismatch', (1,) * 1024, 'a' * 64, 2),
                           8, initial_cache=mismatch)
    assert events == ['pack', 'release']


@pytest.mark.parametrize('adapter', [True, None])
def test_actual_native_wrapper_rejects_changed_or_unproven_attention(adapter):
    loaded, calls, _, _ = native_fixture((17,))
    loaded.provenance['raw_prefill_adapter'] = adapter
    with pytest.raises(QualificationError, match='unchanged attention'):
        generate_reference(loaded, RequestSpec('bad-native', (1,) * 1024, 'd' * 64, 1), 8)
    assert calls == []


@pytest.mark.parametrize('options,message', [({'corrupt_layer': 31}, 'cache length'), ({'finite': False}, 'nonfinite')])
def test_actual_native_wrapper_rejects_last_layer_length_and_nonfinite(options, message):
    loaded, calls, syncs, _ = native_fixture((17,), **options)
    with pytest.raises(QualificationError, match=message):
        generate_reference(loaded, RequestSpec('invalid-native', (1,) * 1024, 'e' * 64, 1), 32)
    assert len(calls) == 1 and syncs == ['sync']


@pytest.mark.parametrize('field,value', [('decode_forwards', 4), ('final_cache_length', 1030)])
def test_generation_comparison_checks_forward_count_and_final_length(field, value):
    expected = GenerationRecord((17, 19), 'length', 1, 1025)
    values = dict(token_ids=expected.token_ids, stop_reason=expected.stop_reason,
                  decode_forwards=expected.decode_forwards, final_cache_length=expected.final_cache_length)
    values[field] = value
    assert compare_generation(expected, GenerationRecord(**values)) == {
        'kind': field, 'expected': getattr(expected, field), 'actual': value}


def test_busy_cli_lock_stops_before_gpu_imports_and_output_creation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    lock_path, output = tmp_path / 'hardware.lock', tmp_path / 'results'
    # A fresh process proves collection state cannot hide a prior GPU import.
    code = """
import importlib.abc, runpy, sys
class RejectGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers', 'triton') or fullname == 'kvolt_runtime.backend.model':
            raise AssertionError('GPU import attempted before busy-lock rejection: ' + fullname)
sys.meta_path.insert(0, RejectGPU())
sys.argv = ['qualify_backend.py'] + sys.argv[1:]
runpy.run_path('tools/qualify_backend.py', run_name='__main__')
"""
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, '-c', code, '--suite', 'smoke',
            '--snapshot', str(tmp_path / 'absent-snapshot'), '--output', str(output),
            '--lock', str(lock_path)], cwd=root, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert 'GPU hardware lock busy' in result.stderr
    assert 'GPU import attempted' not in result.stderr
    assert not output.exists()


def test_existing_output_is_preserved_without_gpu_imports(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'results'
    output.mkdir()
    evidence = output / 'first-mismatch.json'
    original = json.dumps({'phase': 'generation', 'case': 'first-failure'})
    evidence.write_text(original)
    result = subprocess.run([sys.executable, 'tools/qualify_backend.py', '--suite', 'smoke',
        '--snapshot', str(tmp_path / 'absent-snapshot'), '--output', str(output),
        '--lock', str(tmp_path / 'unused.lock')], cwd=root, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1 and 'output must be empty' in result.stderr
    assert evidence.read_text() == original
    assert not (tmp_path / 'unused.lock').exists()


@pytest.mark.parametrize('existing', [True, False])
def test_hardware_lock_preserves_readonly_inode_and_excludes_other_process(tmp_path, existing):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('readonly_lock_tool', root / 'tools/qualify_backend.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    path = tmp_path / 'hardware.lock'
    if existing:
        path.write_bytes(b'existing shared lock record')
        path.chmod(0o444)
        before = path.stat()
    script = """
import fcntl, sys
with open(sys.argv[1], 'rb') as stream:
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(3)
"""
    with tool.hardware_lock(path):
        inode = path.stat().st_ino
        child = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, timeout=15)
        assert child.returncode == 3, child.stderr
        if existing:
            assert inode == before.st_ino
            assert path.stat().st_mode == before.st_mode
            assert path.read_bytes() == b'existing shared lock record'
    assert path.stat().st_ino == inode
    child = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, timeout=15)
    assert child.returncode == 0, child.stderr


def test_attention_fixture_converts_parameters_without_touching_rotary_buffers():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('b2_gpu_fixture',root/'tests/integration/conftest.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Data:
        def __init__(self,dtype): self.dtype=dtype
        def to(self,*,dtype): return Data(dtype)
    class Parameter:
        def __init__(self): self.data=Data('float32')
        def is_floating_point(self): return True
    parameters=[Parameter(),Parameter()]
    rotary=Data('float32')
    attention=SimpleNamespace(parameters=lambda:iter(parameters),rotary_emb=SimpleNamespace(inv_freq=rotary))
    torch=SimpleNamespace(no_grad=nullcontext,float16='float16')
    module.fp16_parameters_only(attention,torch)
    assert all(parameter.data.dtype=='float16' for parameter in parameters)
    assert attention.rotary_emb.inv_freq is rotary
    assert attention.rotary_emb.inv_freq.dtype=='float32'


def test_device_report_uses_structured_properties_without_repr():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('b2_qualification_tool',root/'tools/qualify_backend.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Properties:
        name='Orin'
        major=8
        minor=7
        total_memory=32*1024**3
        multi_processor_count=16
        def __str__(self):
            raise UnicodeDecodeError('utf-8',b'\x86',0,1,'invalid start byte')
        __repr__=__str__
    torch=SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda index:Properties()),
                          version=SimpleNamespace(cuda='12.6'))
    report=module.device_report(torch)
    assert report=={'name':'Orin','major':8,'minor':7,'total_memory':32*1024**3,
                    'multi_processor_count':16,'torch_cuda':'12.6'}
    assert json.loads(json.dumps(report,allow_nan=False))==report
