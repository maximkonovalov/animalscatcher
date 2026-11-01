"""Tests for ac.MotionDetector -- the per-camera frame-differencing check
that decides which frames the AI gets to look at.

Frames are 640x480, so every pixel of the detector's 160x120 comparison
image is a 4x4 block here; changes below are drawn in whole blocks so their
size in the comparison image is exact.
"""
import numpy as np
import pytest

import ac


def solid_frame(value, shape=(480, 640, 3)):
    return np.full(shape, value, dtype="uint8")


def with_block(frame, x, y, w, h, value=250):
    out = frame.copy()
    out[y:y + h, x:x + w] = value
    return out


def make(threshold=0.02, pixel_delta=25, blur=1, lookback=0.0):
    # blur=1 (off) keeps the changed area equal to what the tests draw.
    return ac.MotionDetector(threshold, pixel_delta, lookback, blur)


def test_first_frame_has_nothing_to_compare_with_and_is_not_motion():
    detector = make()

    moved, size = detector.check(solid_frame(100), now=0)

    assert moved is False
    assert size == 0.0


def test_identical_frame_is_not_motion():
    detector = make()
    frame = solid_frame(100)
    detector.check(frame, now=0)

    moved, size = detector.check(frame.copy(), now=0.25)

    assert moved is False
    assert size == 0.0


def test_large_change_is_motion_and_reports_its_size():
    detector = make()
    frame = solid_frame(100)
    detector.check(frame, now=0)

    # A 200x200 block of a 640x480 frame is 13.0% of it.
    moved, size = detector.check(with_block(frame, 100, 100, 200, 200), now=0.25)

    assert moved is True
    assert size == pytest.approx(200 * 200 / (640 * 480), abs=0.002)


def test_the_size_is_a_fraction_of_the_frame_whatever_its_resolution():
    detector = make()
    frame = solid_frame(100, (720, 1280, 3))
    detector.check(frame, now=0)

    # The same 13.0% as above, on a 1280x720 frame (one comparison pixel
    # is 8x6 of its pixels, and the block is aligned to that grid).
    moved, size = detector.check(with_block(frame, 200, 198, 400, 300), now=0.25)

    assert moved is True
    assert size == pytest.approx(400 * 300 / (1280 * 720), abs=0.002)


def test_a_change_the_pixel_delta_does_not_reach_is_not_motion():
    frame = solid_frame(100)
    subtle = with_block(frame, 100, 100, 200, 200, value=115)  # 15 levels

    strict, sensitive = make(pixel_delta=25), make(pixel_delta=10)
    for detector in (strict, sensitive):
        detector.check(frame, now=0)

    assert strict.check(subtle, now=0.25)[0] is False
    assert sensitive.check(subtle, now=0.25)[0] is True


def test_threshold_controls_how_small_a_change_counts():
    frame = solid_frame(100)
    small = with_block(frame, 100, 100, 40, 40)  # 0.5% of the frame

    sensitive, strict = make(threshold=0.001), make(threshold=0.02)
    for detector in (sensitive, strict):
        detector.check(frame, now=0)

    assert sensitive.check(small, now=0.25)[0] is True
    assert strict.check(small, now=0.25)[0] is False


def test_scattered_changes_do_not_add_up_to_motion():
    # 30 separate one-pixel changes cover 0.16% of the comparison image
    # between them, over the 0.1% threshold -- but no single patch comes
    # near it. Only the largest connected patch counts, so noise doesn't.
    frame = solid_frame(100)
    speckled = frame.copy()
    for i in range(5):
        for j in range(6):
            speckled[j * 16:j * 16 + 4, i * 16:i * 16 + 4] = 250
    detector = make(threshold=0.001)
    detector.check(frame, now=0)

    moved, size = detector.check(speckled, now=0.25)

    assert moved is False
    assert size == pytest.approx(1 / (160 * 120))


def test_tiny_noise_is_not_motion_at_the_default_blur():
    frame = solid_frame(100)
    noisy = frame.copy()
    noisy[0:2, 0:2] = 105
    detector = ac.MotionDetector(0.02, 25, 0.0, ac.MOTION_BLUR)
    detector.check(frame, now=0)

    assert detector.check(noisy, now=0.25)[0] is False


def moving_block(step):
    """A 40x40 block that has moved 4 pixels (one in the comparison image)
    per step."""
    return with_block(solid_frame(100), 40 + 4 * step, 100, 40, 40)


def run_moving_block(detector, steps=12, spacing=0.25):
    return [detector.check(moving_block(i), now=i * spacing)[0]
            for i in range(steps)]


def test_a_slow_mover_is_missed_against_the_previous_frame():
    # Each step shifts the block one comparison pixel: only a thin strip
    # at each edge differs from the frame before, well under the threshold.
    results = run_moving_block(make(threshold=0.002, lookback=0.0))

    assert not any(results)


def test_a_longer_lookback_makes_the_same_slow_mover_visible():
    # Compared with the frame two seconds earlier, the block has moved
    # eight pixels, and the strips it left and entered are eight wide.
    results = run_moving_block(make(threshold=0.002, lookback=2.0))

    assert any(results)


def test_until_the_history_is_lookback_long_the_oldest_frame_is_the_reference():
    detector = make(lookback=10.0)
    quiet = solid_frame(100)
    detector.check(quiet, now=0)
    visitor = with_block(quiet, 100, 100, 200, 200)
    detector.check(visitor, now=0.25)

    # Nothing is ten seconds old yet, so this still compares with the
    # first (empty) frame, not with the identical one just before it.
    assert detector.check(visitor.copy(), now=0.5)[0] is True


def test_with_no_lookback_the_reference_is_the_previous_check():
    detector = make(lookback=0.0)
    quiet = solid_frame(100)
    detector.check(quiet, now=0)
    visitor = with_block(quiet, 100, 100, 200, 200)
    detector.check(visitor, now=0.25)

    assert detector.check(visitor.copy(), now=0.5)[0] is False


def test_history_does_not_grow_without_bound():
    detector = make(lookback=2.0)
    for i in range(400):
        detector.check(solid_frame(100), now=i * 0.25)

    # 2 s at one check per 0.25 s is 8 frames, plus the reference itself.
    assert len(detector._history) <= 10


def test_reset_forgets_the_earlier_frames():
    detector = make()
    quiet = solid_frame(100)
    detector.check(quiet, now=0)
    detector.reset()

    # No baseline any more, so even a very different frame is not motion.
    moved, size = detector.check(with_block(quiet, 0, 0, 640, 480), now=0.25)

    assert (moved, size) == (False, 0.0)
