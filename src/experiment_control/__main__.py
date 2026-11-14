"""Wrap a foreground command without editing its source; no shell evaluation."""
import argparse
import json
import os
import signal
import subprocess
import sys

from .contracts import EnvironmentError
from .profile import environment_from_profile


class _Interrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'received {signal.Signals(signum).name} ({signum})')


def _terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def foreground(command, *, checkpoint=None):
    """Wait for the foreground process; terminate its group on interruption.

    Commands must not daemonize or leave detached descendants. Such processes
    cannot be tracked by a generic scope and are outside its completion contract.
    """
    if not command or any(type(arg) is not str or '\x00' in arg for arg in command):
        raise ValueError('command must be a nonempty argument list')
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            try:
                return process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                if checkpoint is not None:
                    checkpoint()
    finally:
        _terminate(process)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output-dir', required=True, help='new artifact directory; never overwritten')
    parser.add_argument('--warmup-command-json', help='optional JSON argv array; runs outside workload scope')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('supply a foreground command after --')
    previous = {}
    env = None

    try:
        warmup_command = json.loads(args.warmup_command_json) if args.warmup_command_json else None
        if warmup_command is not None and (type(warmup_command) is not list or not warmup_command):
            raise ValueError('warmup command must be a nonempty JSON argv array')
        def warmup():
            code = foreground(warmup_command)
            if code:
                raise subprocess.CalledProcessError(code, warmup_command)
        env = environment_from_profile(args.profile, output_dir=args.output_dir,
                                       warmup=warmup if warmup_command is not None else None)
        def interrupted(signum, _frame):
            raise _Interrupted(signum)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        with env:
            code = foreground(command, checkpoint=env.checkpoint)
            if code:
                raise subprocess.CalledProcessError(code, command)
        return 0
    except _Interrupted as error:
        return 128 + error.signum
    except subprocess.CalledProcessError as error:
        return 128 - error.returncode if error.returncode < 0 else min(error.returncode, 255)
    except (EnvironmentError, ValueError, OSError, TypeError) as error:
        print(f'experiment-control: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    raise SystemExit(main())
