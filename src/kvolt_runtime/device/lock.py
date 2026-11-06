"""Use the existing experimental lock inode; never wait or replace it."""
import fcntl
import os
from pathlib import Path

from ..execution.contracts import PreparationError

COMMON_LOCK = Path('/run/lock/kvolt-p3-hardware.lock')


class HardwareLock:
    def __init__(self, path=COMMON_LOCK):
        self.path = Path(path)
        self._file = None
        self._pid = None

    @property
    def held(self):
        return self._file is not None and not self._file.closed and self._pid == os.getpid()

    def __enter__(self):
        if self.held:
            raise PreparationError('HARDWARE_LOCK_REENTRY', 'device_lock', 'hardware lock already held')
        try:
            handle = self.path.open('rb')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            self._file, self._pid = handle, os.getpid()
            self.assert_held()
        except BlockingIOError as error:
            raise PreparationError('HARDWARE_BUSY', 'device_lock', 'hardware lock busy; no waiting permitted') from error
        except OSError as error:
            self.__exit__(None, None, None)
            raise PreparationError('HARDWARE_LOCK_UNAVAILABLE', 'device_lock', 'existing hardware lock unavailable') from error
        except PreparationError:
            self.__exit__(None, None, None)
            raise
        return self

    def assert_held(self):
        if not self.held:
            raise PreparationError('HARDWARE_LOCK_REQUIRED', 'device_lock', 'caller must hold hardware lock')
        try:
            opened, current = os.fstat(self._file.fileno()), self.path.stat()
        except OSError as error:
            raise PreparationError('HARDWARE_LOCK_REPLACED', 'device_lock', 'hardware lock inode unavailable') from error
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise PreparationError('HARDWARE_LOCK_REPLACED', 'device_lock', 'hardware lock inode changed')

    def __exit__(self, *_):
        handle, self._file, self._pid = self._file, None, None
        if handle is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()
