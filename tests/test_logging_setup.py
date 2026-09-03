"""Tests for the logging setup added in v0.10: propagation, and the
PID file / SIGUSR1 reopen handler that let newsyslog rotate the
launchd-redirected stdout/stderr files without a daemon restart.

The actual os.dup2() reopen behavior in _handle_log_reopen() isn't
exercised here -- hijacking fd 1/2 in-process would break pytest's own
output capture. That mechanic was verified manually via a subprocess
that rotates its own log file and confirms a SIGUSR1 signal makes
subsequent writes land in the fresh file, not the archived one; see
RELEASES.md."""
import os

import ac


def test_logger_does_not_propagate_to_root():
    # PytorchWildlife pulls in ultralytics, which installs its own
    # StreamHandler(stderr) on the root logger at import time. Without
    # propagate=False, every message logged here would also be
    # duplicated, unformatted, into stderr via that unrelated handler --
    # found by inspecting logging.getLogger().handlers after `import
    # ac`, not by guessing.
    assert ac.logger.propagate is False


def test_write_pid_file_writes_current_pid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    ac._write_pid_file()

    pid_file = tmp_path / "ac.pid"
    assert pid_file.read_text() == str(os.getpid())


def test_handle_log_reopen_is_noop_without_env_vars(monkeypatch):
    # A plain `python3 ac.py` run (no launchd, no AC_STDOUT_LOG/
    # AC_STDERR_LOG) should tolerate SIGUSR1 harmlessly rather than
    # erroring out trying to reopen paths it was never told about.
    monkeypatch.delenv("AC_STDOUT_LOG", raising=False)
    monkeypatch.delenv("AC_STDERR_LOG", raising=False)

    ac._handle_log_reopen(0, None)
