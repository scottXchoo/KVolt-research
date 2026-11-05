"""Pinned source and installed binary checks without importing GPU packages."""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
from types import MappingProxyType

from ..errors import QualificationError

BASELINE_COMMIT = 'ae3c07e6c1a89d0c5c637ab0a035f954b7a3792c'
KIVI_COMMIT = '876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6'
MODEL_SHA256 = 'd17fa4a0b808525c4165b6a83d10c1cbf8efe99395163f84cef113fd71a71e9b'
SOURCE_HASHES = {
    'models/llama_kivi.py': MODEL_SHA256,
    'quant/new_pack.py': '3678af0e34a0ba18e5d80a4128acf11d4070667c800a15540a16d07253a4f75e',
    'quant/matmul.py': 'e212072490b99f3e157a04c985f5bbaae4028fb14581de4e1a5029f212ec22bd',
    'quant/csrc/gemv_cuda.cu': 'b60f440322ab8859aa3fd429e1441dc5d7010bd913b9119c450f1237a24bf4a4',
}
VERSIONS = {
    'torch': '2.5.0a0+872d972e41.nv24.8', 'transformers': '4.43.1',
    'flash-attn': '2.7.4.post1', 'triton': '3.5.0', 'accelerate': '0.33.0',
}
BINARY_HASHES = {
    'kivi-gemv': ('kivi_gemv', 'e6349ddbef539cead78d9600db7a19c7a107fb12eb05dcd5085096ef034a320b'),
    'flash-attn': ('flash_attn_2_cuda', 'd0c8ed4c5e3b933913dcdd368e52996b7116133fedbc4b7e43d072d4e2c958a6'),
}
# Device names select only a previously inspected build. They do not detect a
# device, authorize arbitrary hashes, or change shared source/model/version pins.
NATIVE_BUILD_PROFILES = MappingProxyType({
    'agx1': MappingProxyType(dict(BINARY_HASHES)),
    'agx2': MappingProxyType({
        'kivi-gemv': ('kivi_gemv', '59a8cc13a82859612494d3fde0554edd2eee4cdcb7b9cdbb4f1a98297fb9e27a'),
        'flash-attn': ('flash_attn_2_cuda', '10e5657f58c462cbc052b1798792e94aab498b47d039bad5ddc305be0a414c11'),
    }),
})
MODEL_FILES = {
    'config.json': '29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e',
    'generation_config.json': '189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd',
    'tokenizer_config.json': '177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424',
    'tokenizer.json': '79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4',
    'model-00001-of-00004.safetensors': '2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668',
    'model-00002-of-00004.safetensors': '09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15',
    'model-00003-of-00004.safetensors': 'fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa',
    'model-00004-of-00004.safetensors': '92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b',
}


def execution_environment():
    """Capture only the process fields needed to reproduce native-library loading."""
    return {'python_executable': sys.executable,
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH')}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def checked_file(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or digest(path) != expected:
        raise QualificationError(f'pinned file hash mismatch: {path}')
    return {'path': str(path), 'sha256': expected}


def native_build_hashes(native_profile='agx1'):
    if not isinstance(native_profile, str) or native_profile not in NATIVE_BUILD_PROFILES:
        raise QualificationError(f'unknown native build profile: {native_profile!r}')
    return NATIVE_BUILD_PROFILES[native_profile]


def validate_sources(kivi_root, snapshot, *, native_profile='agx1'):
    """Reject changed files/dependencies before torch import or model loading."""
    binary_hashes = native_build_hashes(native_profile)
    root, snapshot = Path(kivi_root).resolve(), Path(snapshot).resolve()
    vendor = Path(__file__).with_name('vendor')
    sources = {'vendor_model': checked_file(vendor / 'llama_kivi.py', MODEL_SHA256)}
    sources['vendor_license'] = checked_file(vendor / 'LICENSE.KIVI',
        'add9ea09ab043685efec19b2ba456d6889b62a128ff43e2eca5d37489fdbf8a8')
    for relative, expected in SOURCE_HASHES.items():
        sources[relative] = checked_file(root / relative, expected)
    versions = {}
    for package, expected in VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise QualificationError(f'missing pinned dependency: {package}') from error
        if actual != expected:
            raise QualificationError(f'{package}: expected {expected}, got {actual}')
        versions[package] = actual
    binaries = {}
    for package, (module, expected) in binary_hashes.items():
        distribution = importlib.metadata.distribution(package)
        files = [distribution.locate_file(p) for p in distribution.files or []
                 if Path(p).name.startswith(module + '.') and str(p).endswith('.so')]
        if len(files) != 1:
            raise QualificationError(f'cannot uniquely locate pinned native module: {module}')
        binaries[module] = checked_file(files[0], expected)
    model_files = {name: checked_file(snapshot / name, sha) for name, sha in MODEL_FILES.items()}
    return {'native_profile': native_profile,
            'baseline_commit': BASELINE_COMMIT, 'kivi_upstream_commit': KIVI_COMMIT,
            'sources': sources, 'versions': versions, 'binaries': binaries,
            'model_files': model_files, 'snapshot': str(snapshot),
            'execution_environment': execution_environment()}


@contextmanager
def source_import_path(kivi_root):
    previous = list(sys.path)
    sys.path.insert(0, str(Path(kivi_root).resolve()))
    try:
        yield
    finally:
        sys.path[:] = previous


def import_pinned_source(kivi_root, provenance):
    """Candidate and reference use this same unchanged model source module."""
    with source_import_path(kivi_root):
        source = importlib.import_module('kvolt_runtime.backend.vendor.llama_kivi')
        for name, relative in [('quant.new_pack', 'quant/new_pack.py'), ('quant.matmul', 'quant/matmul.py')]:
            module = importlib.import_module(name)
            actual = Path(module.__file__).resolve()
            expected = Path(kivi_root, relative).resolve()
            if actual != expected:
                raise QualificationError(f'unexpected import origin for {name}: {actual}')
            checked_file(actual, SOURCE_HASHES[relative])
        origins = {}
        for name, record in provenance['binaries'].items():
            module = importlib.import_module(name)
            actual = Path(module.__file__).resolve()
            if actual != Path(record['path']):
                raise QualificationError(f'unexpected import origin for {name}: {actual}')
            checked_file(actual, record['sha256'])
            origins[name] = str(actual)
        origins.update({name: str(Path(sys.modules[name].__file__).resolve())
                        for name in ['quant.new_pack', 'quant.matmul', 'torch', 'transformers',
                                     'transformers.models.llama.modeling_llama']})
        origins['model'] = str(Path(source.__file__).resolve())
        checked_file(origins['model'], MODEL_SHA256)
        provenance['import_origins'] = origins
        provenance['import_source_sha256'] = {name: digest(path) for name, path in origins.items()}
        return source
