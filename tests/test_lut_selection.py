"""The canonical selection binds real versioned sources without changing them."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from kvolt_runtime.policy import selection as module
from kvolt_runtime.policy.compact import load_lut_repository
from kvolt_runtime.policy.selection import DEFAULT_LUT_SELECTION, load_current_lut_repository
from kvolt_runtime.policy.types import LUTError
from kvolt_runtime.qualification.contracts import _lut_binding

LUTS = Path(__file__).resolve().parents[1] / 'data/luts'
RUNTIME = 'agx2-runtime-repeat5'
REFERENCE = 'agx2-reference-repeat10'
RUNTIME_PIN = 'b421fcd10ecb57df7d6962aa9a5f48e264e90defc0c76a3544218d31685f783b'
REFERENCE_PIN = 'd332875ccce626dd99700da8c7a128bf4801577333e76ee40a987f4051311e40'


@pytest.fixture
def copied(tmp_path):
    root = tmp_path / 'luts'
    for name in ('current', RUNTIME, REFERENCE):
        shutil.copytree(LUTS / name, root / name)
    return root / 'current/selection.json'


def rewrite(path, update):
    value = json.loads(path.read_bytes())
    update(value)
    path.write_text(json.dumps(value))


def test_default_selection_is_fixed_independent_of_cwd(tmp_path, monkeypatch):
    assert DEFAULT_LUT_SELECTION == LUTS / 'current/selection.json'
    monkeypatch.chdir(tmp_path)
    repo = load_current_lut_repository()
    assert repo.provenance.kind == 'measured'
    assert repo.provenance.manifest_sha256 == RUNTIME_PIN
    metadata = json.loads(repo.provenance.metadata_json)
    assert metadata['target_repository']['manifest_sha256'] == REFERENCE_PIN
    assert metadata['target_repository']['source_conditions']['technical_repeats_per_length'] == 10
    assert repo.prefill_target(1024) == 0.817132383
    assert repo.decode_target(8192) == 0.13676398722700586
    assert metadata['current_selection']['path'] == str(DEFAULT_LUT_SELECTION)
    assert metadata['current_selection']['sha256'] == hashlib.sha256(DEFAULT_LUT_SELECTION.read_bytes()).hexdigest()


def test_default_keeps_exact_component_and_qualification_identities():
    default = load_current_lut_repository()
    explicit = load_lut_repository(LUTS / RUNTIME, reference_lut_dir=LUTS / REFERENCE)
    assert _lut_binding(default.provenance) == _lut_binding(explicit.provenance)
    old = load_lut_repository(LUTS / RUNTIME, reference_lut_dir=LUTS / 'agx2-reference-repeat5')
    assert _lut_binding(default.provenance) != _lut_binding(old.provenance)
    assert default.provenance.files_sha256 == explicit.provenance.files_sha256


def test_canonical_directory_is_also_an_explicit_loader_entry_point():
    default = load_current_lut_repository()
    explicit = load_lut_repository(LUTS / 'current')
    assert _lut_binding(default.provenance) == _lut_binding(explicit.provenance)
    with pytest.raises(LUTError, match='reference-lut-dir'):
        load_lut_repository(LUTS / 'current', reference_lut_dir=LUTS / REFERENCE)


def test_selection_cannot_recurse_or_mix_distribution_formats(copied):
    original = copied.read_bytes()
    rewrite(copied, lambda value: value['runtime'].update(directory='current'))
    with pytest.raises(LUTError, match='itself'):
        load_current_lut_repository(copied)
    copied.write_bytes(original)
    (copied.parent / 'prefill.json').write_text('[]')
    with pytest.raises(LUTError, match='ambiguous'):
        load_lut_repository(copied.parent)


@pytest.mark.parametrize('component', ['runtime', 'reference'])
def test_selector_hash_must_match_loaded_component(copied, component):
    rewrite(copied, lambda value: value[component].update(manifest_sha256='0' * 64))
    with pytest.raises(LUTError, match='SHA-256'):
        load_current_lut_repository(copied)


@pytest.mark.parametrize('component', ['runtime', 'reference'])
@pytest.mark.parametrize('name', ['..', '.', '../agx2-runtime-repeat5', '/tmp/escape', 'nested/name', 'a\\b', ''])
def test_selector_does_not_escape_sibling_distributions(copied, component, name):
    rewrite(copied, lambda value: value[component].update(directory=name))
    with pytest.raises(LUTError):
        load_current_lut_repository(copied)


@pytest.mark.parametrize('value', ['0' * 63, 'G' * 64, None, 1, True])
def test_selector_requires_real_digest(copied, value):
    rewrite(copied, lambda data: data['reference'].update(manifest_sha256=value))
    with pytest.raises(LUTError):
        load_current_lut_repository(copied)


@pytest.mark.parametrize('change', [
    lambda value: value.update(schema='unknown'),
    lambda value: value.pop('reference'),
    lambda value: value.update(extra='unused'),
    lambda value: value.update(runtime=[]),
    lambda value: value['runtime'].update(extra='unused'),
])
def test_selector_rejects_incomplete_or_ambiguous_contract(copied, change):
    rewrite(copied, change)
    with pytest.raises(LUTError):
        load_current_lut_repository(copied)


@pytest.mark.parametrize('content', [b'{}', b'[]', b'{"schema": NaN}', b'{"schema":1,"schema":2}', b'not-json'])
def test_selector_rejects_invalid_json_or_structure(copied, content):
    copied.write_bytes(content)
    with pytest.raises(LUTError):
        load_current_lut_repository(copied)


def test_missing_default_does_not_search_cwd_or_fall_back(copied, monkeypatch):
    copied.unlink()
    monkeypatch.setattr(module, 'DEFAULT_LUT_SELECTION', copied)
    with pytest.raises(LUTError, match='missing or unreadable'):
        load_current_lut_repository()


@pytest.mark.parametrize('component,filename', [(RUNTIME, 'prefill.json'), (REFERENCE, 'reference.json'),
                                               (REFERENCE, 'source-validation.json')])
def test_canonical_selection_preserves_component_integrity_checks(copied, component, filename):
    path = copied.parent.parent / component / filename
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(LUTError, match='SHA-256'):
        load_current_lut_repository(copied)


def test_selector_and_distribution_symlinks_rejected(copied, tmp_path):
    linked = tmp_path / 'selection.json'
    linked.symlink_to(copied)
    with pytest.raises(LUTError, match='symlink'):
        load_current_lut_repository(linked)
    reference = copied.parent.parent / REFERENCE
    reference.rename(reference.with_name('saved-reference'))
    reference.symlink_to(reference.with_name('saved-reference'), target_is_directory=True)
    with pytest.raises(LUTError, match='symlink|escapes'):
        load_current_lut_repository(copied)
