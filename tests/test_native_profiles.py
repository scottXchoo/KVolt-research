"""CPU-only gates for explicitly pinned native builds; no hardware detection."""
import importlib.abc
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvolt_runtime.backend import model, provenance as p
from kvolt_runtime.config import RuntimeConfig
from kvolt_runtime.errors import QualificationError
from kvolt_runtime.qualification.reference import GenerationRecord

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'agx1': {
        'kivi-gemv': ('kivi_gemv', 'e6349ddbef539cead78d9600db7a19c7a107fb12eb05dcd5085096ef034a320b'),
        'flash-attn': ('flash_attn_2_cuda', 'd0c8ed4c5e3b933913dcdd368e52996b7116133fedbc4b7e43d072d4e2c958a6'),
    },
    'agx2': {
        'kivi-gemv': ('kivi_gemv', '59a8cc13a82859612494d3fde0554edd2eee4cdcb7b9cdbb4f1a98297fb9e27a'),
        'flash-attn': ('flash_attn_2_cuda', '10e5657f58c462cbc052b1798792e94aab498b47d039bad5ddc305be0a414c11'),
    },
}


def installed_build(tmp_path, monkeypatch, installed):
    """Fake only discovery/non-native files; real checked_file gates native hashes."""
    original = p.checked_file
    paths = {}
    hashes = {}
    for package, (module, sha) in EXPECTED[installed].items():
        path = tmp_path / (module + '.test.so')
        path.write_bytes(b'fixture native binary')
        paths[package] = path
        hashes[path] = sha
    monkeypatch.setattr(p, 'digest', lambda path: hashes[Path(path)])
    monkeypatch.setattr(p, 'checked_file', lambda path, expected: original(path, expected)
                        if str(path).endswith('.so') else {'path': str(path), 'sha256': expected})
    monkeypatch.setattr(p.importlib.metadata, 'version', lambda package: p.VERSIONS[package])
    monkeypatch.setattr(p.importlib.metadata, 'distribution', lambda package: SimpleNamespace(
        files=[paths[package]], locate_file=lambda path: path))
    return paths, hashes


def test_known_profiles_are_immutable_and_agx1_pins_unchanged():
    assert p.BINARY_HASHES == EXPECTED['agx1']
    assert set(p.NATIVE_BUILD_PROFILES) == {'agx1', 'agx2'}
    for name in EXPECTED:
        assert p.native_build_hashes(name) == EXPECTED[name]
        with pytest.raises(TypeError):
            p.NATIVE_BUILD_PROFILES[name]['kivi-gemv'] = ('arbitrary', '0' * 64)
    with pytest.raises(TypeError):
        p.NATIVE_BUILD_PROFILES['unregistered'] = {}


@pytest.mark.parametrize('selected', [None, 'agx1', 'agx2'])
def test_selected_exact_native_build_is_recorded(tmp_path, monkeypatch, selected):
    installed_build(tmp_path, monkeypatch, selected or 'agx1')
    kwargs = {} if selected is None else {'native_profile': selected}
    result = p.validate_sources(tmp_path / 'kivi', tmp_path / 'snapshot', **kwargs)
    assert result['native_profile'] == (selected or 'agx1')
    assert {module: record['sha256'] for module, record in result['binaries'].items()} == {
        module: sha for module, sha in EXPECTED[selected or 'agx1'].values()}


@pytest.mark.parametrize('selected,installed', [('agx1', 'agx2'), ('agx2', 'agx1')])
def test_cross_profile_hashes_fail_before_gpu_import(tmp_path, monkeypatch, selected, installed):
    installed_build(tmp_path, monkeypatch, installed)
    class RejectGPU(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('torch', 'transformers', 'triton'):
                raise AssertionError('GPU import before native hash check')
    monkeypatch.setattr(sys, 'meta_path', [RejectGPU()] + sys.meta_path)
    with pytest.raises(QualificationError, match='hash mismatch'):
        model.load_model(tmp_path / RuntimeConfig().snapshot, RuntimeConfig(),
                         kivi_root=tmp_path / 'kivi', native_profile=selected)


@pytest.mark.parametrize('profile', ['agx1', 'agx2'])
@pytest.mark.parametrize('package', ['kivi-gemv', 'flash-attn'])
def test_either_native_binary_mismatch_fails_before_gpu_import(tmp_path, monkeypatch, profile, package):
    paths, hashes = installed_build(tmp_path, monkeypatch, profile)
    hashes[paths[package]] = '0' * 64
    class RejectGPU(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('torch', 'transformers', 'triton'):
                raise AssertionError('GPU import before native hash check')
    monkeypatch.setattr(sys, 'meta_path', [RejectGPU()] + sys.meta_path)
    with pytest.raises(QualificationError, match=paths[package].name):
        model.load_model(tmp_path / RuntimeConfig().snapshot, RuntimeConfig(),
                         kivi_root=tmp_path / 'kivi', native_profile=profile)


@pytest.mark.parametrize('name', ['AGX2', 'auto', '', None, {}])
def test_unknown_profile_rejected_without_discovery_or_gpu_import(tmp_path, monkeypatch, name):
    def forbidden(*args):
        raise AssertionError('profile must fail before file/dependency discovery')
    monkeypatch.setattr(p, 'checked_file', forbidden)
    with pytest.raises(QualificationError, match='native build profile'):
        model.load_model(tmp_path / RuntimeConfig().snapshot, RuntimeConfig(), native_profile=name)
    with pytest.raises(QualificationError, match='native build profile'):
        model.InferenceBackend(native_profile=name)


@pytest.mark.parametrize('selected', [None, 'agx2'])
def test_backend_profile_reaches_default_loader_and_preserves_injected_signature(monkeypatch, tmp_path, selected):
    calls = []
    loaded = model.LoadedModel(object(), object(), object(), {})
    def default(path, config, **kwargs):
        calls.append(kwargs)
        return loaded
    monkeypatch.setattr(model, 'load_model', default)
    monkeypatch.setenv('KVOLT_NATIVE_PROFILE', 'agx2')
    kwargs = {} if selected is None else {'native_profile': selected}
    model.InferenceBackend(kivi_root=tmp_path, **kwargs).load_session(tmp_path, RuntimeConfig())
    assert calls == [{'kivi_root': tmp_path, 'native_profile': selected or 'agx1'}]
    injected_calls = []
    def injected(path, config):
        injected_calls.append((path, config))
        return loaded
    model.InferenceBackend(loader=injected, native_profile='agx2').load_session(tmp_path, RuntimeConfig())
    assert len(injected_calls) == 1


def test_backend_cli_uses_same_explicit_profile_for_reference_and_candidate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('profile_qualification_tool', ROOT / 'tools/qualify_backend.py')
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    from kvolt_runtime.qualification import compare, reference
    from kvolt_runtime.contracts import RequestSpec
    request = RequestSpec('fixture', (128000, 42), 'a' * 64, 2)
    case = SimpleNamespace(key='fixture-g32-n2', request=request, group=32)
    monkeypatch.setattr(compare, 'load_cohort', lambda *args: ([], {}))
    monkeypatch.setattr(compare, 'build_cases', lambda *args: [case])
    monkeypatch.setattr(compare, 'bounded_chunks', lambda *args: [[case]])
    monkeypatch.setattr(q, 'hardware_lock', lambda path: nullcontext())
    monkeypatch.setattr(q.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(q, 'device_report', lambda torch: {})
    monkeypatch.setattr(reference, 'generate_reference', lambda *args, **kwargs: GenerationRecord((128009,), 'eos', 0, 2))
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
        q.main(['--suite', 'smoke', '--snapshot', str(tmp_path), '--kivi-root', str(tmp_path / 'kivi'),
                '--output', str(tmp_path / 'output'), '--native-profile', 'agx2'])
    assert [kind for kind, kwargs in calls] == ['reference', 'candidate']
    assert all(kwargs['native_profile'] == 'agx2' for kind, kwargs in calls)
    import json
    assert json.loads((tmp_path / 'output/plan.json').read_text())['native_profile'] == 'agx2'


@pytest.mark.parametrize('tool', ['qualify_backend.py', 'qualify_first_eos.py'])
def test_cli_help_and_unknown_profile_are_gpu_free(tool, tmp_path):
    code = """
import importlib.abc, runpy, sys
class RejectGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers', 'triton') or fullname == 'kvolt_runtime.backend.model':
            raise AssertionError('GPU/model import attempted: ' + fullname)
sys.meta_path.insert(0, RejectGPU())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-c', code, 'tools/' + tool, '--help'],
                            cwd=ROOT, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert '--native-profile {agx1,agx2}' in result.stdout
    args = ['--snapshot', str(tmp_path), '--output', str(tmp_path / 'out'), '--native-profile', 'auto']
    if tool == 'qualify_backend.py':
        args += ['--suite', 'smoke']
    result = subprocess.run([sys.executable, '-c', code, 'tools/' + tool, *args],
                            cwd=ROOT, text=True, capture_output=True, timeout=15)
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr and 'GPU/model import attempted' not in result.stderr
