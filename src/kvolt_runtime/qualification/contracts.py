"""CPU-only, immutable identities for scoped M4 qualification evidence.

Qualification approves a recorded configuration, never an energy/SLO outcome.
The sealed LUT's original eligibility flags remain untouched.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

from ..errors import ContractError

SCHEMA = 'kvolt-runtime-qualification-v1'
REQUIRED_CHECKS = ('cpu_policy', 'data', 'backend_exact', 'timing_power',
                   'device_control', 'failure_recovery', 'reproducibility', 'warmup')
POLICY_VERSIONS = {'numpy': '1.26.4', 'scipy': '1.15.3', 'kneed': '0.9.0'}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError('duplicate qualification JSON key: ' + key)
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        canonical_json(value).encode('utf-8')
        return value
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ContractError('qualification evidence must be finite UTF-8 JSON') from error


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def qualification_binding(*, repo, device_profile, power_profile, native_profile, runtime):
    """Read identifiers without GPU/scientific imports or any device mutation.

    DeviceController repeats physical identity/preparation checks under the
    common lock. Backend independently validates the actual model/kernel bytes.
    """
    from ..backend.provenance import MODEL_FILES, SOURCE_HASHES, native_build_hashes, VERSIONS
    from ..config import SNAPSHOT
    if native_profile != 'agx2' or repo.provenance.kind != 'measured':
        raise ContractError('formal qualification requires measured AGX2 data and native profile')
    if repo.provenance.source_device.casefold() != 'agx2' or power_profile is None:
        raise ContractError('formal qualification requires AGX2 LUTs and collected power')
    serial = device_profile.identity_path.read_text(encoding='utf-8').rstrip('\x00\n').strip()
    identity = hashlib.sha256(serial.encode('utf-8')).hexdigest()
    boot = device_profile.boot_id_path.read_text(encoding='utf-8').strip()
    if not serial or identity != device_profile.identity_expected or not boot:
        raise ContractError('qualification hardware identity is missing or mismatched')
    if runtime.get('source_worktree_dirty') is not False or not runtime.get('source_sha256'):
        raise ContractError('formal qualification requires a clean identified runtime source')
    required = runtime.get('required_policy_versions')
    if required != POLICY_VERSIONS or any(runtime.get('installed_distribution_versions', {}).get(name) != version
                                             for name, version in required.items()):
        raise ContractError('formal qualification requires pinned installed policy dependencies')
    return {
        'runtime_source_sha256': runtime['source_sha256'],
        'python': runtime['python'], 'platform': runtime['platform'],
        'python_executable': sys.executable, 'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH'),
        'policy_versions': runtime['required_policy_versions'],
        'native_profile': native_profile,
        'native_binaries': {key: {'module': module, 'sha256': sha}
                            for key, (module, sha) in native_build_hashes(native_profile).items()},
        'backend_versions': dict(VERSIONS), 'backend_sources': dict(SOURCE_HASHES),
        'model_snapshot': SNAPSHOT, 'model_files': dict(MODEL_FILES),
        'lut': {'kind': repo.provenance.kind, 'source_device': repo.provenance.source_device,
                'source_commit': repo.provenance.source_commit,
                'manifest_sha256': repo.provenance.manifest_sha256,
                'files_sha256': dict(repo.provenance.files_sha256)},
        'identity': {'physical_id_sha256': identity, 'boot_id': boot},
        'device_profile': device_profile.as_dict(), 'power_profile': power_profile.to_dict(),
    }


@dataclass(frozen=True)
class QualificationGrant:
    """Validated evidence token passed internally; no CLI boolean can create it."""
    binding_json: str
    coverage_json: str
    report_sha256: str
    report_path: str

    @property
    def binding(self):
        return strict_json(self.binding_json)

    @property
    def coverage(self):
        return strict_json(self.coverage_json)

    def assert_requests(self, prepared):
        coverage = self.coverage
        for item in prepared.requests:
            maximum = coverage.get(str(item.spec.length))
            if type(maximum) is not int or not 1 <= item.spec.nmax <= maximum <= 512:
                raise ContractError('request length/output cap is outside qualified coverage')

    def assert_device(self, snapshot):
        if snapshot.get('verified') is not True or snapshot.get('identity') != self.binding['identity']:
            raise ContractError('locked device identity differs from M4 qualification')

    def as_dict(self):
        return {'schema_version': SCHEMA, 'report_path': self.report_path,
                'report_sha256': self.report_sha256,
                'binding_sha256': value_sha256(self.binding), 'coverage': self.coverage}
