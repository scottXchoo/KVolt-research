"""Qualification is scoped execution evidence, not an SLO acceptance test."""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kvolt_runtime.errors import ContractError
from kvolt_runtime.qualification.contracts import (
    REQUIRED_CHECKS, SCHEMA, canonical_json, value_sha256,
)
from kvolt_runtime.qualification.evidence import load_evidence


@pytest.fixture
def bundle(tmp_path):
    binding = {
        'runtime_source_sha256': 'a' * 64, 'native_profile': 'agx2',
        'identity': {'physical_id_sha256': 'b' * 64, 'boot_id': 'boot-a'},
        'lut': {'kind': 'measured', 'source_device': 'AGX2',
                'manifest_sha256': 'c' * 64, 'runtime_eligible': False,
                'runtime_validation_required': True},
        'model_files': {'model': 'd' * 64},
        'native_binaries': {'kernel': 'e' * 64},
        'device_profile': {'timeout': 3}, 'power_profile': {'collector': 'ina3221'},
    }
    coverage = {'1024': 1, '4096': 128, '8192': 512}
    summaries = {}
    raw = tmp_path / 'raw.log'
    raw.write_text('executed checks: 3 passed\nSLO violations: 2\n')
    report = {'schema_version': SCHEMA, 'binding': binding, 'coverage': coverage,
              'checks': {}, 'artifacts': {'raw.log': hashlib.sha256(raw.read_bytes()).hexdigest()}}
    for check in REQUIRED_CHECKS:
        summaries[check] = {
            'schema_version': 'kvolt-qualification-check-v1', 'check': check,
            'binding_sha256': value_sha256(binding), 'coverage': copy.deepcopy(coverage),
            'command': ['python', '-m', 'audit', '--check', check], 'exit_code': 0,
            'tests': {'run': 3, 'passed': 3, 'failed': 0, 'skipped': 0},
            'artifacts': ['raw.log'],
        }
        report['checks'][check] = {'status': 'passed', 'artifact': check + '.json'}
    path = tmp_path / 'qualification.json'

    def save():
        for check, summary in summaries.items():
            artifact = tmp_path / (check + '.json')
            artifact.write_text(canonical_json(summary))
            report['artifacts'][artifact.name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        path.write_text(canonical_json(report))
    save()
    return SimpleNamespace(path=path, binding=binding, report=report,
                           summaries=summaries, save=save)


def test_valid_grant_scoped_immutable_and_slo_agnostic(bundle):
    original = copy.deepcopy(bundle.binding)
    grant = load_evidence(bundle.path, bundle.binding)
    assert grant.binding == original
    assert grant.coverage == bundle.report['coverage']
    assert grant.report_sha256 == hashlib.sha256(bundle.path.read_bytes()).hexdigest()
    assert grant.report_path == str(bundle.path)
    bundle.binding['lut']['runtime_eligible'] = True
    assert grant.binding['lut']['runtime_eligible'] is False
    assert original['lut']['runtime_validation_required'] is True
    request = lambda length, nmax: SimpleNamespace(spec=SimpleNamespace(length=length, nmax=nmax))
    grant.assert_requests(SimpleNamespace(requests=[request(1024, 1), request(8192, 512)]))
    with pytest.raises(ContractError):
        grant.assert_requests(SimpleNamespace(requests=[request(2048, 1)]))
    with pytest.raises(ContractError):
        grant.assert_requests(SimpleNamespace(requests=[request(1024, 2)]))


@pytest.mark.parametrize('key', ['runtime_source_sha256', 'native_profile', 'model_files',
                                  'native_binaries', 'lut', 'identity',
                                  'device_profile', 'power_profile'])
def test_stale_binding_denied(bundle, key):
    expected = copy.deepcopy(bundle.binding)
    expected[key] = 'changed'
    with pytest.raises(ContractError, match='binding'):
        load_evidence(bundle.path, expected)


@pytest.mark.parametrize('coverage', [{}, {'1023': 1}, {'8193': 1}, {'01024': 1},
    {'1e3': 1}, {'1024': True}, {'1024': 1.0}, {'1024': '1'}, {'1024': 0},
    {'1024': 513}, {'1024': None}, [], {'+1024': 1}, {' 1024': 1}, {'１０２４': 1}])
def test_invalid_coverage_denied(bundle, coverage):
    bundle.report['coverage'] = coverage
    bundle.save()
    with pytest.raises(ContractError, match='coverage'):
        load_evidence(bundle.path, bundle.binding)


@pytest.mark.parametrize('mutation', ['missing_check', 'failed_check', 'unknown_check',
    'unknown_field', 'unknown_check_field', 'unreferenced', 'no_outputs',
    'wrong_check', 'wrong_binding', 'wrong_coverage', 'nonzero_exit',
    'no_command', 'all_skipped', 'failed_test', 'bool_count', 'count_mismatch',
    'gpu_skipped', 'summary_unknown_field', 'duplicate_output', 'summary_as_output'])
def test_incomplete_or_false_claims_denied(bundle, mutation):
    summary = bundle.summaries['backend_exact']
    if mutation == 'missing_check':
        del bundle.report['checks']['warmup']
    elif mutation == 'failed_check':
        bundle.report['checks']['warmup']['status'] = 'failed'
    elif mutation == 'unknown_check':
        bundle.report['checks']['extra'] = {'status': 'passed', 'artifact': 'raw.log'}
    elif mutation == 'unknown_field':
        bundle.report['approved'] = True
    elif mutation == 'unknown_check_field':
        bundle.report['checks']['warmup']['approved'] = True
    elif mutation == 'unreferenced':
        bundle.report['artifacts']['orphan.log'] = 'a' * 64
    elif mutation == 'no_outputs':
        summary['artifacts'] = []
    elif mutation == 'wrong_check':
        summary['check'] = 'data'
    elif mutation == 'wrong_binding':
        summary['binding_sha256'] = '0' * 64
    elif mutation == 'wrong_coverage':
        summary['coverage']['8192'] = 511
    elif mutation == 'nonzero_exit':
        summary['exit_code'] = 1
    elif mutation == 'no_command':
        summary['command'] = []
    elif mutation == 'all_skipped':
        summary['tests'] = {'run': 3, 'passed': 0, 'failed': 0, 'skipped': 3}
    elif mutation == 'failed_test':
        summary['tests'] = {'run': 3, 'passed': 2, 'failed': 1, 'skipped': 0}
    elif mutation == 'bool_count':
        summary['tests']['passed'] = True
    elif mutation == 'count_mismatch':
        summary['tests']['run'] = 4
    elif mutation == 'gpu_skipped':
        summary['tests'] = {'run': 4, 'passed': 3, 'failed': 0, 'skipped': 1}
    elif mutation == 'summary_unknown_field':
        summary['allow_approximate'] = True
    elif mutation == 'duplicate_output':
        summary['artifacts'].append('raw.log')
    elif mutation == 'summary_as_output':
        summary['artifacts'] = ['backend_exact.json']
    bundle.save()
    with pytest.raises(ContractError):
        load_evidence(bundle.path, bundle.binding)


def test_allowed_skips_are_preserved(bundle):
    for check in ('cpu_policy', 'data', 'failure_recovery'):
        bundle.summaries[check]['tests'] = {'run': 106, 'passed': 3, 'failed': 0, 'skipped': 103}
    bundle.save()
    assert load_evidence(bundle.path, bundle.binding).coverage['8192'] == 512


@pytest.mark.parametrize('target', ['report', 'summary', 'raw'])
def test_duplicate_or_nonfinite_json_denied(bundle, target):
    path = bundle.path if target == 'report' else bundle.path.parent / 'backend_exact.json'
    path.write_text('{"x":1,"x":2}' if target != 'raw' else '{"x":1e999}')
    if target != 'report':
        bundle.report['artifacts'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        bundle.path.write_text(canonical_json(bundle.report))
    with pytest.raises(ContractError):
        load_evidence(bundle.path, bundle.binding)


@pytest.mark.parametrize('target', ['backend_exact.json', 'raw.log'])
def test_tampered_artifact_denied(bundle, target):
    (bundle.path.parent / target).write_bytes(b'tampered')
    with pytest.raises(ContractError, match='hash'):
        load_evidence(bundle.path, bundle.binding)


@pytest.mark.parametrize('relative', ['../escape.log', '/etc/passwd', './raw.log',
    'sub/../raw.log', 'sub//raw.log', 'sub\\raw.log'])
def test_noncanonical_or_escaping_artifact_denied(bundle, relative):
    bundle.report['artifacts'][relative] = bundle.report['artifacts'].pop('raw.log')
    for summary in bundle.summaries.values():
        summary['artifacts'] = [relative]
    bundle.save()
    with pytest.raises(ContractError):
        load_evidence(bundle.path, bundle.binding)


@pytest.mark.parametrize('target', ['raw.log', 'backend_exact.json', 'qualification.json'])
def test_symlink_file_denied(bundle, target):
    original = bundle.path.parent / target
    moved = bundle.path.parent / ('moved-' + target)
    original.rename(moved)
    original.symlink_to(moved.name)
    with pytest.raises(ContractError):
        load_evidence(bundle.path, bundle.binding)


def test_symlink_root_ancestor_denied(bundle, tmp_path):
    alias = tmp_path / 'alias'
    alias.symlink_to(bundle.path.parent, target_is_directory=True)
    with pytest.raises(ContractError):
        load_evidence(alias / bundle.path.name, bundle.binding)


def test_missing_empty_and_nonregular_outputs_denied(bundle):
    raw = bundle.path.parent / 'raw.log'
    raw.unlink()
    with pytest.raises(ContractError):
        load_evidence(bundle.path, bundle.binding)
    raw.touch()
    bundle.report['artifacts']['raw.log'] = hashlib.sha256(b'').hexdigest()
    bundle.path.write_text(canonical_json(bundle.report))
    with pytest.raises(ContractError):
        load_evidence(bundle.path, bundle.binding)
    raw.unlink()
    os.mkfifo(raw)
    with pytest.raises(ContractError):
        load_evidence(bundle.path, bundle.binding)


def test_cpu_only_import_without_scientific_or_gpu_modules():
    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','scipy','kneed','transformers','triton'}:
            raise RuntimeError('forbidden module: ' + fullname)
sys.meta_path.insert(0, Block())
from kvolt_runtime.qualification.evidence import load_evidence
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=environment)
    assert result.returncode == 0, result.stderr
