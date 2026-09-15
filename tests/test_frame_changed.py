"""Tests for ac._frame_changed -- the cheap per-raw-frame motion check
added so camera_thread can queue a frame immediately when something
moves, rather than relying solely on the fixed frame_interval cadence
(which can miss a fast animal's entire visible window between samples).
"""
import numpy as np

import ac


def solid_frame(value, shape=(480, 640, 3)):
    return np.full(shape, value, dtype="uint8")


def test_first_call_with_no_previous_frame_always_reports_changed():
    changed, gray = ac._frame_changed(None, solid_frame(100), 0.02)

    assert changed is True
    assert gray is not None


def test_identical_frame_is_not_reported_as_changed():
    frame = solid_frame(100)
    _, gray = ac._frame_changed(None, frame, 0.02)

    changed, _ = ac._frame_changed(gray, frame.copy(), 0.02)

    assert changed is False


def test_large_change_is_reported_as_motion():
    frame = solid_frame(100)
    _, gray = ac._frame_changed(None, frame, 0.02)

    moved = frame.copy()
    moved[100:300, 100:300] = 250  # a large, bright block -- simulated animal

    changed, _ = ac._frame_changed(gray, moved, 0.02)

    assert changed is True


def test_tiny_noise_is_not_reported_as_motion():
    frame = solid_frame(100)
    _, gray = ac._frame_changed(None, frame, 0.02)

    noisy = frame.copy()
    noisy[0:2, 0:2] = 105  # a couple of pixels, subtly different

    changed, _ = ac._frame_changed(gray, noisy, 0.02)

    assert changed is False


def test_threshold_controls_sensitivity():
    frame = solid_frame(100)
    _, gray = ac._frame_changed(None, frame, 0.02)

    moved = frame.copy()
    moved[100:150, 100:150] = 250  # a small, but real, change

    # A very low threshold should catch it...
    changed_sensitive, _ = ac._frame_changed(gray, moved, 0.001)
    # ...while a very high one should not.
    changed_strict, _ = ac._frame_changed(gray, moved, 0.5)

    assert changed_sensitive is True
    assert changed_strict is False
