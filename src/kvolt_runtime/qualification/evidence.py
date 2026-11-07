"""Fail-closed validation of scoped M4 reports, using only the standard library.

Reports bind machine check summaries and their original outputs by SHA-256.
This verifies the recorded evidence and scope; hashes are provenance, not a
signature or independent proof that an untrusted report author ran a command.
Commands are recorded strings and are never executed by this loader. Original
LUT eligibility flags and observed SLO violations have no approval semantics.
See docs/qualification-evidence.md for the machine-write format.
"""
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat

from ..errors import ContractError
from .contracts import (
    REQUIRED_CHECKS, SCHEMA, QualificationGrant, canonical_json,
    strict_json, value_sha256,
)

CHECK_SCHEMA = 'kvolt-qualification-check-v1'
_JSON_LIMIT = 16 * 1024 * 1024
_SKIPPABLE = frozenset(('cpu_policy', 'data', 'failure_recovery'))


def _exact(value, fields, description):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ContractError(description + ' has missing or unknown fields')


def _same(left, right):
    # Python equality alone conflates booleans/integers and integers/floats.
    return canonical_json(left) == canonical_json(right)


def _coverage(value):
    if not isinstance(value, dict) or not value:
        raise ContractError('qualification coverage must be a nonempty length-to-maxN mapping')
    for length, maximum in value.items():
        if (not isinstance(length, str) or not re.fullmatch(r'[1-9][0-9]{3}', length)
                or not 1024 <= int(length) <= 8192
                or type(maximum) is not int or not 1 <= maximum <= 512):
            raise ContractError('qualification coverage requires canonical lengths 1024..8192 and maxN 1..512')


def _relative(value):
    if (not isinstance(value, str) or not value or '\\' in value or '\x00' in value
            or PurePosixPath(value).is_absolute()
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ContractError('qualification artifact path must stay within the report root')
    return value.split('/')


def _digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ContractError('qualification artifact hash must be lowercase SHA-256')


def _open_root(path):
    """Open each absolute directory without following any symlink ancestor."""
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read(root, relative, *, json_document=False, expected_digest=None):
    """Hash the exact bytes read, traversing only directories pinned by fds."""
    components = _relative(relative)
    directory = os.dup(root)
    try:
        for component in components[:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
        with os.fdopen(descriptor, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ContractError('qualification artifact must be a regular file')
            digest = hashlib.sha256()
            chunks = []
            size = 0
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
                if json_document:
                    if size > _JSON_LIMIT:
                        raise ContractError('qualification JSON exceeds 16 MiB')
                    chunks.append(chunk)
            if not size:
                raise ContractError('qualification artifacts must not be empty')
            if expected_digest is not None and digest.hexdigest() != expected_digest:
                raise ContractError('qualification artifact hash mismatch: ' + relative)
            document = strict_json(b''.join(chunks).decode('utf-8')) if json_document else None
            return digest.hexdigest(), document
    finally:
        os.close(directory)


def _summary(value, check, binding, coverage, summaries):
    _exact(value, ('schema_version', 'check', 'binding_sha256', 'coverage',
                   'command', 'exit_code', 'tests', 'artifacts'), 'qualification check summary')
    if value['schema_version'] != CHECK_SCHEMA or value['check'] != check:
        raise ContractError('qualification check summary schema/name mismatch')
    if value['binding_sha256'] != value_sha256(binding):
        raise ContractError('qualification check summary binding mismatch')
    _coverage(value['coverage'])
    if not _same(value['coverage'], coverage):
        raise ContractError('qualification check summary coverage mismatch')
    command = value['command']
    if (not isinstance(command, list) or not command
            or any(not isinstance(part, str) or not part.strip() or '\x00' in part for part in command)):
        raise ContractError('qualification check must record its executed command')
    if type(value['exit_code']) is not int or value['exit_code'] != 0:
        raise ContractError('qualification check command did not succeed')
    tests = value['tests']
    _exact(tests, ('run', 'passed', 'failed', 'skipped'), 'qualification test counts')
    if (any(type(count) is not int or count < 0 for count in tests.values())
            or tests['passed'] < 1 or tests['failed'] != 0
            or tests['run'] != tests['passed'] + tests['failed'] + tests['skipped']
            or (tests['skipped'] and check not in _SKIPPABLE)):
        raise ContractError('qualification checks require executed passing tests and no failures')
    outputs = value['artifacts']
    if not isinstance(outputs, list) or not outputs:
        raise ContractError('qualification check requires original machine output artifacts')
    seen = set()
    for output in outputs:
        _relative(output)
        if output in seen or output in summaries:
            raise ContractError('qualification check output must be a distinct raw artifact')
        seen.add(output)
    return seen


def load_evidence(path, binding):
    """Return an immutable grant only for complete, matching, hashed evidence.

    ``binding`` must be the current trusted ``qualification_binding`` result.
    Coverage lists explicit input lengths; it never implies interpolation to
    unlisted lengths. An entry maxN qualifies only caps at or below that value.
    Missing/stale evidence, failed checks and unsafe paths raise ContractError.
    """
    root = None
    try:
        if not isinstance(binding, dict) or not binding:
            raise ContractError('qualification requires the current nonempty binding')
        # Reject non-JSON/nonfinite input and detach the trusted caller's object.
        binding = strict_json(canonical_json(binding))
        report_path = Path(os.path.abspath(os.fspath(path)))
        root = _open_root(report_path.parent)
        report_sha256, report = _read(root, report_path.name, json_document=True)
        _exact(report, ('schema_version', 'binding', 'coverage', 'checks', 'artifacts'),
               'qualification report')
        if report['schema_version'] != SCHEMA:
            raise ContractError('qualification report schema mismatch')
        if not _same(report['binding'], binding):
            raise ContractError('qualification report binding mismatch')
        _coverage(report['coverage'])
        _exact(report['checks'], REQUIRED_CHECKS, 'qualification required checks')
        artifacts = report['artifacts']
        if not isinstance(artifacts, dict) or not artifacts:
            raise ContractError('qualification report requires hashed artifacts')
        for relative, digest in artifacts.items():
            _relative(relative)
            _digest(digest)
            if relative == report_path.name:
                raise ContractError('qualification report cannot be its own artifact')
        summaries = set()
        for check, receipt in report['checks'].items():
            _exact(receipt, ('status', 'artifact'), 'qualification check receipt')
            if receipt['status'] != 'passed':
                raise ContractError('qualification required check did not pass: ' + check)
            _relative(receipt['artifact'])
            if receipt['artifact'] not in artifacts or receipt['artifact'] in summaries:
                raise ContractError('qualification check needs its own hashed summary')
            summaries.add(receipt['artifact'])
        referenced = set(summaries)
        for check, receipt in report['checks'].items():
            relative = receipt['artifact']
            _, summary = _read(root, relative, json_document=True,
                               expected_digest=artifacts[relative])
            referenced.update(_summary(summary, check, binding, report['coverage'], summaries))
        if referenced != set(artifacts):
            raise ContractError('qualification has missing or unreferenced artifact claims')
        for relative in referenced - summaries:
            _read(root, relative, expected_digest=artifacts[relative])
        return QualificationGrant(canonical_json(binding), canonical_json(report['coverage']),
                                  report_sha256, str(report_path))
    except ContractError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise ContractError('qualification evidence is unreadable or invalid: ' + str(error)) from error
    finally:
        if root is not None:
            os.close(root)
