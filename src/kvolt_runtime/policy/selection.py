"""One canonical, explicit selection of the versioned prediction/target sources."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .compact import load_lut_repository
from .lut import _json, _read
from .types import LUTError


# Resolve from the source checkout, so execution never depends on the shell cwd.
DEFAULT_LUT_SELECTION = Path(__file__).resolve().parents[3] / 'data/luts/current/selection.json'


def _component(root, spec, selection_dir):
    if type(spec) is not dict or set(spec) != {'directory', 'manifest_sha256'}:
        raise LUTError('current LUT component requires directory and manifest_sha256')
    name, digest = spec['directory'], spec['manifest_sha256']
    if (type(name) is not str or not name or name in ('.', '..')
            or any(character in name for character in ('/', '\\', '\x00'))):
        raise LUTError('current LUT component directory must be a single sibling name')
    if (type(digest) is not str or len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)):
        raise LUTError('current LUT component manifest SHA-256 must be a lowercase digest')
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise LUTError('current LUT component is a symlink or escapes data/luts')
    if path.resolve() == selection_dir:
        raise LUTError('current LUT selection cannot refer to itself')
    return path


def load_current_lut_repository(selection_path=None):
    """Load exactly the selected pinned sources, or fail before hardware access.

    The selection is mutable configuration. Both component manifests and all
    source files still pass their existing frozen loader contracts. Historical
    exports remain available through explicit paths; no latest-file search or
    synthetic fallback takes place.
    """
    path = Path(DEFAULT_LUT_SELECTION if selection_path is None else selection_path)
    if path.is_symlink():
        raise LUTError('current LUT selection must not be a symlink')
    data = _read(path)
    selection = _json(data, 'selection.json')
    if (type(selection) is not dict or set(selection) != {'schema', 'runtime', 'reference'}
            or selection['schema'] != 'kvolt.lut-selection.v1'):
        raise LUTError('invalid current LUT selection schema or components')
    path = path.resolve()
    root = path.parent.parent
    runtime = _component(root, selection['runtime'], path.parent)
    reference = _component(root, selection['reference'], path.parent)
    repository = load_lut_repository(runtime, reference_lut_dir=reference)
    if repository.provenance.manifest_sha256 != selection['runtime']['manifest_sha256']:
        raise LUTError('current LUT runtime manifest SHA-256 mismatch')
    metadata = json.loads(repository.provenance.metadata_json)
    if metadata['target_repository']['manifest_sha256'] != selection['reference']['manifest_sha256']:
        raise LUTError('current LUT reference manifest SHA-256 mismatch')
    metadata['current_selection'] = {
        'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
        'selection': selection,
    }
    repository.provenance = replace(
        repository.provenance,
        metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
    )
    return repository
