"""CPU-only JSON evidence and exclusively owned, atomic session storage.

Runner owns statuses, validity and timestamps. Storage serializes those fields
without deriving new outcomes, retrying a request, or rereading input files.
"""
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import secrets
import stat

from ..config import require_int
from ..errors import ContractError


def to_json_native(value):
    """Convert ordinary Python evidence without importing tensor/array packages."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError('recorded JSON numbers must be finite')
        return value
    if isinstance(value, Enum):
        return to_json_native(value.value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_json_native(getattr(value, field.name)) for field in fields(value)}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ContractError('recorded JSON object keys must be strings')
        return {key: to_json_native(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [to_json_native(item) for item in value]
    raise ContractError(f'unsupported recorded JSON type: {type(value).__name__}')


def json_dumps(value):
    return json.dumps(to_json_native(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def json_line(value):
    """One complete JSONL record; embedded text newlines remain escaped."""
    return json_dumps(value) + '\n'


def request_record(prepared, *, index=None):
    """Prepared input identity only; no generation/timing values are fabricated."""
    spec = prepared.spec
    result = {'request_id': spec.request_id, 'attempt': 1,
              'prompt_file': str(prepared.prompt.path), 'prompt_sha256': spec.prompt_sha256,
              'token_ids_sha256': spec.token_sha256, 'token_ids': list(spec.token_ids),
              'length': spec.length, 'nmax': spec.nmax, 'metadata': prepared.metadata,
              'input_path': f'inputs/{spec.prompt_sha256}.txt'}
    if index is not None:
        result['index'] = index
    return result


class SessionRecorder:
    """Create a new session directory; never reuse an existing path.

    Every write uses a same-directory temporary, file fsync, atomic replacement
    and parent fsync. Directory file descriptors and O_NOFOLLOW keep writes
    within the claimed tree. A failed write raises its original storage error;
    the caller decides how to report missing paths through stdout/stderr.
    """

    def __init__(self, output_dir):
        # Do not resolve the last component: a dangling symlink is a conflict.
        self.output_dir = Path(os.path.abspath(os.fspath(output_dir)))
        self.output_dir.mkdir(exist_ok=False)
        observed = self.output_dir.lstat()
        self._identity = (observed.st_dev, observed.st_ino)

    def _root_fd(self):
        fd = os.open(self.output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        observed = os.fstat(fd)
        if (observed.st_dev, observed.st_ino) != self._identity:
            os.close(fd)
            raise ContractError('session directory ownership changed')
        return fd

    @staticmethod
    def _parts(relative_path):
        path = Path(relative_path)
        if path.is_absolute() or not path.parts or any(part in ('..', '.') for part in path.parts):
            raise ContractError('artifact path must stay within the session')
        return path.parts

    def _parent_fd(self, relative_path, *, create):
        parts = self._parts(relative_path)
        fd = self._root_fd()
        try:
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, dir_fd=fd)
                        os.fsync(fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise

    def path(self, relative_path):
        """Relative artifact path only when an actual regular file exists."""
        try:
            fd, name = self._parent_fd(relative_path, create=False)
        except FileNotFoundError:
            return None
        try:
            observed = os.stat(name, dir_fd=fd, follow_symlinks=False)
            return str(Path(relative_path)) if stat.S_ISREG(observed.st_mode) else None
        except FileNotFoundError:
            return None
        finally:
            os.close(fd)

    @property
    def session_path(self):
        return self.path('session.json')

    def write_bytes(self, relative_path, value):
        if type(value) is not bytes:
            raise ContractError('artifact bytes must be bytes')
        fd, name = self._parent_fd(relative_path, create=True)
        temporary = None
        try:
            try:
                old = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                old = None
            if old is not None and not stat.S_ISREG(old.st_mode):
                raise ContractError('artifact destination must be a regular file')
            temporary = f'.{name}.{secrets.token_hex(16)}.tmp'
            file_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                              0o600, dir_fd=fd)
            with os.fdopen(file_fd, 'wb') as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
                published_identity = os.fstat(stream.fileno())
            os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
            temporary = None
            os.fsync(fd)
            # An open descriptor still refers to a renamed directory. Reopen
            # the public path before claiming success, so a safe write into an
            # unlinked/moved directory cannot yield a nonexistent artifact link.
            public_fd, public_name = self._parent_fd(relative_path, create=False)
            try:
                public_parent, original_parent = os.fstat(public_fd), os.fstat(fd)
                public_file = os.stat(public_name, dir_fd=public_fd, follow_symlinks=False)
                if ((public_parent.st_dev, public_parent.st_ino) !=
                        (original_parent.st_dev, original_parent.st_ino) or
                        (public_file.st_dev, public_file.st_ino) !=
                        (published_identity.st_dev, published_identity.st_ino)):
                    raise ContractError('artifact directory ownership changed during write')
            finally:
                os.close(public_fd)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary, dir_fd=fd)
                except FileNotFoundError:
                    pass
            os.close(fd)
        return str(Path(relative_path))

    def write_text(self, relative_path, value):
        if type(value) is not str:
            raise ContractError('artifact text must be a string')
        return self.write_bytes(relative_path, value.encode('utf-8', errors='strict'))

    def atomic_json(self, relative_path, value):
        return self.write_text(relative_path, json_line(value))

    def save_preflight(self, prepared):
        original = None
        if prepared.manifest_raw is not None:
            original = self.write_bytes('manifest.original.json', prepared.manifest_raw)
        resolved = self.atomic_json('manifest.resolved.json', prepared.resolved_manifest)
        requests = []
        for index, request in enumerate(prepared.requests):
            record = request_record(request, index=index)
            self.write_bytes(record['input_path'], request.prompt.text.encode('utf-8', errors='strict'))
            requests.append(record)
        preflight = self.atomic_json('preflight.json', {
            'schema_version': 'kvolt-preflight-v1', 'manifest_path': prepared.manifest_path,
            'manifest_sha256': prepared.manifest_sha256, 'requests': requests,
        })
        return {'manifest_original_path': original, 'manifest_resolved_path': resolved,
                'preflight_path': preflight}

    def save_request(self, index, prepared, result, generation=None, output_text=None):
        require_int(index, 'request index', 0)
        base = f'requests/{index:06d}'
        request = self.atomic_json(f'{base}/request.json', request_record(prepared, index=index))
        generation_path = (None if generation is None else
                           self.atomic_json(f'{base}/generation.json', generation))
        output_path = (None if output_text is None else
                       self.write_text(f'{base}/output.txt', output_text))
        result_path = self.atomic_json(f'{base}/result.json', result)
        return {'request_path': request, 'generation_path': generation_path,
                'result_path': result_path, 'output_path': output_path}

    def save_session(self, summary):
        return self.atomic_json('session.json', summary)
