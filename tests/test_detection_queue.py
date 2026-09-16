"""Tests for detection_queue's priority ordering -- motion-triggered
frames (priority 0) should be drained before frame_interval-only ones
(priority 1) whenever both are waiting, regardless of insertion order,
and the seq tie-breaker must prevent PriorityQueue's internal
comparison from ever falling through to comparing frame (a numpy
array), which raises ValueError on an ambiguous truth value."""
import numpy as np
import pytest

import ac


@pytest.fixture(autouse=True)
def drain_queue_after_test():
    """detection_queue is a module-level singleton shared across tests;
    make sure a failed assertion doesn't leave items behind for the
    next test."""
    yield
    while not ac.detection_queue.empty():
        ac.detection_queue.get_nowait()


def fake_frame(value=0):
    return np.full((5, 5, 3), value, dtype="uint8")


def test_motion_triggered_frame_drains_before_earlier_routine_frames():
    ac.detection_queue.put_nowait((1, ac._next_queue_seq(), "cam04", fake_frame(1)))
    ac.detection_queue.put_nowait((1, ac._next_queue_seq(), "cam05", fake_frame(2)))
    ac.detection_queue.put_nowait((0, ac._next_queue_seq(), "cam06", fake_frame(3)))

    priority, _seq, cam_id, _frame = ac.detection_queue.get()

    assert (priority, cam_id) == (0, "cam06")


def test_drain_order_is_priority_first_then_insertion_order():
    ac.detection_queue.put_nowait((1, ac._next_queue_seq(), "a", fake_frame()))
    ac.detection_queue.put_nowait((0, ac._next_queue_seq(), "b", fake_frame()))
    ac.detection_queue.put_nowait((1, ac._next_queue_seq(), "c", fake_frame()))
    ac.detection_queue.put_nowait((0, ac._next_queue_seq(), "d", fake_frame()))

    drained = [ac.detection_queue.get()[2] for _ in range(4)]

    # Both priority-0 items ("b", "d") before both priority-1 items
    # ("a", "c"); within each priority, insertion order (seq) wins.
    assert drained == ["b", "d", "a", "c"]


def test_same_priority_random_frames_never_raise_on_comparison():
    # Regression guard: without the seq tie-breaker, PriorityQueue would
    # fall through to comparing (cam_id, frame) tuples when priority
    # ties, and comparing two numpy arrays with `<` raises ValueError
    # ("truth value of an array... is ambiguous") rather than something
    # cleanly catchable -- this must never happen.
    # detection_queue's real maxsize is small (shared with production),
    # so put/get in batches rather than filling past capacity.
    for _ in range(4):
        for _ in range(10):
            frame = np.random.randint(0, 255, (5, 5, 3), dtype="uint8")
            ac.detection_queue.put_nowait((1, ac._next_queue_seq(), "cam04", frame))
        for _ in range(10):
            ac.detection_queue.get()  # would raise ValueError if this broke


def test_next_queue_seq_is_strictly_increasing():
    seqs = [ac._next_queue_seq() for _ in range(10)]

    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
