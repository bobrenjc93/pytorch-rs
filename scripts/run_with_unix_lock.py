#!/usr/bin/env python3
"""Run a command while holding a cross-platform Unix advisory file lock."""

import argparse
import errno
import fcntl
import os
import signal
import stat
import subprocess
import sys


def _open_lock(path):
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        file_descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise SystemExit(f"refusing symlinked lock file: {path}") from error
        if error.errno == errno.EISDIR:
            raise SystemExit(f"lock path is a directory: {path}") from error
        raise

    try:
        mode = os.fstat(file_descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise SystemExit(f"lock path is not a regular file: {path}")
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(file_descriptor)
        raise
    return file_descriptor


def _run_command(command):
    child = subprocess.Popen(command, start_new_session=True)
    received_signal = None
    previous_handlers = {}

    def forward_signal(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    for signum in (
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGTERM,
    ):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)

    try:
        returncode = child.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if returncode < 0:
        return 128 - returncode
    if received_signal is not None and returncode == 0:
        return 128 + received_signal
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command while holding a Unix advisory file lock.",
    )
    parser.add_argument("lock_path")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()

    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("missing command")

    try:
        lock_fd = _open_lock(arguments.lock_path)
    except OSError as error:
        print(
            f"failed to open lock file {arguments.lock_path}: {error.strerror}",
            file=sys.stderr,
        )
        return 1

    try:
        return _run_command(command)
    except OSError as error:
        print(
            f"failed to exec {command[0]!r} while holding {arguments.lock_path}: "
            f"{error.strerror}",
            file=sys.stderr,
        )
        return 127 if error.errno == errno.ENOENT else 126
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    sys.exit(main())
