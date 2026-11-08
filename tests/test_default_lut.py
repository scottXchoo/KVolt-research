"""CLI defaults use the canonical measured source before model/device setup."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvolt_runtime import cli, run_cli
from kvolt_runtime.policy.types import LUTError

ROOT = Path(__file__).resolve().parents[1]
LUTS = ROOT / 'data/luts'
CURRENT = LUTS / 'current/selection.json'
RUNTIME = LUTS / 'agx2-runtime-repeat5'
REFERENCE = LUTS / 'agx2-reference-repeat10'


def diagnostic_args(*extra):
    return ['diagnose', '--length', '1024', '--max-new-tokens', '1',
            '--pc-elapsed-ns', '0', *extra]


def run_args(tmp_path, *extra):
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('x' * 1024)
    return ['--prompt-file', str(prompt), '--model-dir', str(tmp_path / 'model'),
            '--output-dir', str(tmp_path / 'out'), '--device-profile', str(tmp_path / 'profile'),
            '--max-new-tokens', '1', '--warmup', 'none', *extra]


def assert_current_source(source):
    assert source['kind'] == 'measured' and source['source_device'].casefold() == 'agx2'
    metadata = source['metadata']
    current = metadata['current_selection']
    assert current['path'] == str(CURRENT)
    assert current['sha256'] == hashlib.sha256(CURRENT.read_bytes()).hexdigest()
    assert current['selection'] == json.loads(CURRENT.read_bytes())
    target = metadata['target_repository']
    assert target['manifest_sha256'] == hashlib.sha256((REFERENCE / 'manifest.json').read_bytes()).hexdigest()
    assert target['files_sha256']['reference.json'] == hashlib.sha256((REFERENCE / 'reference.json').read_bytes()).hexdigest()
    assert metadata['compact_manifest']['validation']['performance_runtime_accuracy_or_SLO_validated'] is False
    assert metadata['compact_manifest']['validation']['quality_runtime_eligible'] is False


def test_default_diagnose_from_arbitrary_cwd_without_scientific_imports(tmp_path):
    code = """import json, sys
from kvolt_runtime.cli import main
assert main(sys.argv[1:]) == 0
assert not ({'torch','transformers','triton','flash_attn','numpy','scipy','kneed'} & set(sys.modules))
"""
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    result = subprocess.run([sys.executable, '-c', code, *diagnostic_args()],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert_current_source(report['source'])
    assert report['prefill']['target_s'] == 0.817132383
    assert report['decode']['status'] == 'skipped'
    assert report['predictions_only'] and report['measurements'] is None


def test_default_run_factories_load_measured_before_tokenizer(tmp_path, monkeypatch, capsys):
    from test_run_cli import factories
    events = []
    seam, _ = factories(events)
    repositories = []
    def repository(args):
        events.append('repo')
        repo = run_cli._repository(args)
        repositories.append(repo)
        return repo
    seam = replace(seam, repository=repository)
    monkeypatch.chdir(tmp_path)
    assert run_cli.main(run_args(tmp_path), factories=seam) == 0
    assert events.index('repo') < events.index('tokenizer') < events.index('lock')
    saved = json.loads((tmp_path / 'out/session.json').read_bytes())
    assert_current_source(saved['provenance']['lut'])
    assert saved['provenance']['formal_performance_qualified'] is False
    assert saved['provenance']['lut_device_conditions']['formal_condition_match_qualified'] is False
    assert repositories[0].decode_target(1024) == 0.10339920563534676
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['exit_code'] == 0


@pytest.mark.parametrize('broken', ['missing', 'bad_sha'])
def test_broken_selection_rejected_before_tokenizer_or_hardware(tmp_path, monkeypatch, capsys, broken):
    from kvolt_runtime.policy import selection
    from test_run_cli import factories
    path = tmp_path / 'current/selection.json'
    if broken == 'bad_sha':
        path.parent.mkdir()
        value = json.loads(CURRENT.read_bytes())
        value['runtime']['manifest_sha256'] = '0' * 64
        path.write_text(json.dumps(value))
        for name in ('agx2-runtime-repeat5', 'agx2-reference-repeat10'):
            shutil.copytree(LUTS / name, tmp_path / name)
    monkeypatch.setattr(selection, 'DEFAULT_LUT_SELECTION', path)
    events = []
    seam, _ = factories(events)
    seam = replace(seam, repository=run_cli._repository)
    assert run_cli.main(run_args(tmp_path), factories=seam) == 2
    assert events == []
    record = json.loads(capsys.readouterr().out)
    assert record['errors'][0]['stage'] == 'lut'
    assert cli.main(diagnostic_args()) == 2
    assert json.loads(capsys.readouterr().out)['error']['stage'] == 'lut'


@pytest.mark.parametrize('command', ['run', 'diagnose'])
def test_help_documents_default_and_stays_lightweight(command):
    code = """import sys
from kvolt_runtime.cli import main
try: main([sys.argv[1], '--help'])
except SystemExit as e: assert e.code == 0
assert not ({'torch','transformers','triton','flash_attn','numpy','scipy','kneed'} & set(sys.modules))
"""
    result = subprocess.run([sys.executable, '-c', code, command], text=True, capture_output=True,
                            env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
    assert result.returncode == 0, result.stderr
    assert 'data/luts/current/selection.json' in result.stdout
    assert '--mock' in result.stdout and '--reference-lut-dir' in result.stdout


@pytest.mark.parametrize('builder,option', [(cli._build_repository, 'lut_bundle'), (run_cli._repository, 'lut_dir')])
def test_explicit_mock_and_reference_contracts_preserved(builder, option):
    args = SimpleNamespace(**{option: None}, mock='baseline', reference_lut_dir=None)
    assert builder(args).provenance.kind == 'synthetic'
    args.reference_lut_dir = REFERENCE
    with pytest.raises(ValueError, match='requires --lut-'):
        builder(args)
    args.mock = None
    with pytest.raises(ValueError, match='requires --lut-'):
        builder(args)
    setattr(args, option, RUNTIME)
    args.reference_lut_dir = None
    with pytest.raises(LUTError, match='explicit --reference-lut-dir'):
        builder(args)


@pytest.mark.parametrize('builder,option', [(cli._build_repository, 'lut_bundle'), (run_cli._repository, 'lut_dir')])
@pytest.mark.parametrize('repeats', [3, 5])
def test_explicit_old_reference_selection_preserved(builder, option, repeats):
    reference = LUTS / f'agx2-reference-repeat{repeats}'
    args = SimpleNamespace(**{option: RUNTIME}, mock=None, reference_lut_dir=reference)
    repo = builder(args)
    metadata = json.loads(repo.provenance.metadata_json)
    assert 'current_selection' not in metadata
    assert metadata['target_repository']['manifest_sha256'] == hashlib.sha256((reference / 'manifest.json').read_bytes()).hexdigest()
    assert repo.decode_target(1024) == json.loads((reference / 'reference.json').read_bytes())[0][2]


def test_explicit_sealed_bundle_still_uses_sealed_loader():
    source = Path(os.environ.get('KVOLT_LUT_BUNDLE', ROOT.parent / 'KVolt-lut-source/data/luts/agx2-repeat5-20261103'))
    if not (source / 'bundle-manifest.json').is_file():
        pytest.skip('optional sealed bundle unavailable: set KVOLT_LUT_BUNDLE')
    from kvolt_runtime.policy.lut import MeasuredLUTRepository
    for builder, option in [(cli._build_repository, 'lut_bundle'), (run_cli._repository, 'lut_dir')]:
        repo = builder(SimpleNamespace(**{option: source}, mock=None, reference_lut_dir=None))
        assert type(repo) is MeasuredLUTRepository
        assert 'current_selection' not in json.loads(repo.provenance.metadata_json)


@pytest.mark.parametrize('builder,option', [(cli._build_repository, 'lut_bundle'), (run_cli._repository, 'lut_dir')])
def test_explicit_current_directory_uses_selection_and_rejects_reference_override(builder, option):
    args = SimpleNamespace(**{option: CURRENT.parent}, mock=None, reference_lut_dir=None)
    assert_current_source(cli._source_metadata(builder(args).provenance))
    args.reference_lut_dir = REFERENCE
    with pytest.raises(LUTError, match='cannot override a LUT selection'):
        builder(args)
