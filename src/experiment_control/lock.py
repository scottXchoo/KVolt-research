"""Nonblocking lock on an existing inode, compatible with KVolt HardwareLock."""
import fcntl
import os
from pathlib import Path

from .contracts import EnvironmentError

COMMON_LOCK = '/run/lock/kvolt-p3-hardware.lock'


class FileLock:
    def __init__(self, path=COMMON_LOCK):
        self.path = Path(path)
        self._handle = None
        self._pid = None

    @property
    def held(self):
        return self._handle is not None and not self._handle.closed and self._pid == os.getpid()

    def __enter__(self):
        if self.held:
            raise EnvironmentError('lock reentry is not allowed')
        handle = self.path.open('rb')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._handle, self._pid = handle, os.getpid()
            self.assert_held()
        except BaseException:
            self._handle = self._pid = None
            handle.close()
            raise
        return self

    def assert_held(self):
        if not self.held:
            raise EnvironmentError('hardware lock is not held by this process')
        a, b = os.fstat(self._handle.fileno()), self.path.stat()
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
            raise EnvironmentError('hardware lock inode changed')

    def __exit__(self, *_):
        handle, owner_pid = self._handle, self._pid
        self._handle, self._pid = None, None
        if handle is not None:
            try:
                # fork shares the open-file description. A child must not
                # release its parent's flock by explicitly unlocking it.
                if owner_pid == os.getpid():
                    fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()
