"""Tests for the storage side of the daemon: which files count as snapshots
(_snapshot_files), what the summary's STORAGE block says about them
(_storage_report, _human_size) and cleanup_engine, which deletes from the
same set."""
import datetime
import logging
import os
import time

import pytest

import ac


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """base_output_folder, pointed at a throwaway directory."""
    monkeypatch.setattr(ac, "BASE_OUTPUT_FOLDER", str(tmp_path))
    return tmp_path


@pytest.fixture
def log_lines():
    """What ac's logger writes. It doesn't propagate to the root logger, so
    caplog can't see it."""
    lines = []

    class Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = Collect()
    ac.logger.addHandler(handler)
    yield lines
    ac.logger.removeHandler(handler)


def snapshot(base, cam_id, name, size=10, age_days=0):
    """A `size`-byte file in the camera's folder, `age_days` old."""
    folder = base / cam_id
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_bytes(b"x" * size)
    if age_days:
        then = time.time() - age_days * 86400
        os.utime(path, (then, then))
    return path


def broken_walk():
    """Stands in for _snapshot_files when the storage can't be read."""
    raise PermissionError("denied")
    yield  # makes this a generator, like the real thing


# --- which files are the storage --------------------------------------------

def test_the_snapshots_are_the_files_directly_inside_the_camera_folders(
        storage):
    kept = [snapshot(storage, "cam04", "cam04_Animal_1.jpg"),
            snapshot(storage, "cam04", "cam04_Person_2.jpg"),
            snapshot(storage, "cam06", "cam06_Animal_3.jpg")]
    # Not in a camera folder, a folder rather than a file, and in a folder
    # inside a camera folder: none of these is a snapshot.
    (storage / "stray.jpg").write_bytes(b"x")
    (storage / "cam04" / "older").mkdir()
    (storage / "cam04" / "older" / "old.jpg").write_bytes(b"x")

    assert sorted(ac._snapshot_files()) == sorted(str(p) for p in kept)


def test_a_storage_folder_that_does_not_exist_holds_no_snapshots(
        storage, monkeypatch):
    monkeypatch.setattr(ac, "BASE_OUTPUT_FOLDER", str(storage / "missing"))

    assert list(ac._snapshot_files()) == []


# --- the STORAGE block ------------------------------------------------------

@pytest.mark.parametrize("n, text", [
    (0, "0 B"),
    (1023, "1023 B"),
    (1024, "1.0 KB"),
    (1536, "1.5 KB"),
    (1024 ** 2 - 1, "1.0 MB"),   # not "1024.0 KB"
    (5 * 1024 ** 2, "5.0 MB"),
    (1024 ** 3, "1.0 GB"),
    (3 * 1024 ** 4, "3.0 TB"),
])
def test_sizes_read_in_powers_of_1024(n, text):
    assert ac._human_size(n) == text


def test_the_report_counts_the_files_and_adds_up_their_sizes(storage):
    snapshot(storage, "cam04", "a.jpg", size=1024 ** 2)
    snapshot(storage, "cam04", "b.jpg", size=256 * 1024)
    snapshot(storage, "cam05", "c.jpg", size=256 * 1024)

    assert ac._storage_report() == "- Files: 3\n- Size: 1.5 MB"


def test_an_empty_or_missing_storage_reads_zero(storage, monkeypatch):
    assert ac._storage_report() == "- Files: 0\n- Size: 0 B"

    monkeypatch.setattr(ac, "BASE_OUTPUT_FOLDER", str(storage / "missing"))
    assert ac._storage_report() == "- Files: 0\n- Size: 0 B"


def test_a_file_that_is_gone_by_the_time_it_is_measured_is_not_counted(
        storage, monkeypatch):
    kept = snapshot(storage, "cam04", "kept.jpg", size=2048)
    # Listed, then removed (by cleanup_engine) before it could be measured.
    gone = str(storage / "cam04" / "gone.jpg")
    monkeypatch.setattr(ac, "_snapshot_files",
                        lambda: iter([gone, str(kept)]))

    assert ac._storage_report() == "- Files: 1\n- Size: 2.0 KB"


def test_a_walk_that_fails_is_reported_and_logged_not_raised(
        log_lines, monkeypatch):
    monkeypatch.setattr(ac, "_snapshot_files", broken_walk)

    assert ac._storage_report() == "- unavailable (see the log)"
    assert any("Storage statistics unavailable: denied" in line
               for line in log_lines)


# --- the summary that carries it --------------------------------------------

class _StopSummary(BaseException):
    """Derives from BaseException so it escapes summary_engine's
    `except Exception` and ends the otherwise endless loop."""


def one_summary(monkeypatch):
    """The report summary_engine sends before its second sleep ends the
    test."""
    sent, sleeps = [], []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise _StopSummary

    monkeypatch.setattr(ac, "stats", {
        "Animal": 0, "Person": 0, "Vehicle": 0, "streams": {}, "motion": {},
        "start_time": datetime.datetime(2026, 11, 15, 8, 0)})
    monkeypatch.setattr(ac.time, "sleep", sleep)
    monkeypatch.setattr(ac, "send_telegram_message", sent.append)

    with pytest.raises(_StopSummary):
        ac.summary_engine()

    (report,) = sent
    return report


def test_the_summary_ends_with_the_storage_block(storage, monkeypatch):
    snapshot(storage, "cam04", "a.jpg", size=1024)
    snapshot(storage, "cam05", "b.jpg", size=2048)

    report = one_summary(monkeypatch)

    assert report.endswith("\n\nSTORAGE:\n- Files: 2\n- Size: 3.0 KB")
    assert report.index("AI QUEUE") < report.index("STORAGE:")


def test_the_summary_still_goes_out_when_the_storage_cannot_be_read(
        monkeypatch):
    monkeypatch.setattr(ac, "_snapshot_files", broken_walk)

    report = one_summary(monkeypatch)

    assert "DETECTIONS:" in report
    assert report.endswith("STORAGE:\n- unavailable (see the log)")


# --- cleanup_engine, which deletes from the same set ------------------------

class _StopCleanup(BaseException):
    """Derives from BaseException so it escapes cleanup_engine's
    `except Exception` and ends the otherwise endless loop."""


def test_cleanup_removes_snapshots_older_than_max_age_days_and_only_those(
        storage, log_lines, monkeypatch):
    monkeypatch.setattr(ac, "MAX_AGE_DAYS", 7)
    old = [snapshot(storage, "cam04", "old.jpg", age_days=8),
           snapshot(storage, "cam06", "older.jpg", age_days=30)]
    new = snapshot(storage, "cam04", "new.jpg", age_days=6)
    # Old, but not snapshots: not in a camera folder, and in a folder
    # inside one.
    stray = storage / "stray.jpg"
    stray.write_bytes(b"x")
    nested = storage / "cam04" / "older" / "old.jpg"
    nested.parent.mkdir()
    nested.write_bytes(b"x")
    for path in (stray, nested):
        os.utime(path, (0, 0))

    def sleep(seconds):
        # The sleep after the one pass ends the test.
        raise _StopCleanup

    monkeypatch.setattr(ac.time, "sleep", sleep)
    with pytest.raises(_StopCleanup):
        ac.cleanup_engine()

    assert not any(path.exists() for path in old)
    assert new.exists() and stray.exists() and nested.exists()
    assert "[SYSTEM] Cleanup: Removed 2 old snapshots." in log_lines
