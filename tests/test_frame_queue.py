"""Tests for ac.FrameQueue -- the hand-off from the camera threads to the
single ai_engine thread: one queue of the frames that showed motion, shared
by every camera, oldest first. When it is full the oldest frame is dropped to
make room for the new one."""
import threading

import numpy as np

import ac


def frame(value=0):
    return np.full((5, 5, 3), value, dtype="uint8")


def waiting(queue):
    """Every frame still queued, oldest first."""
    return [f for _, f, _ in queue._frames]


def test_take_returns_the_offered_frame_with_its_camera_and_offer_time(
        monkeypatch):
    monkeypatch.setattr(ac.time, "time", lambda: 1234.5)
    queue = ac.FrameQueue(3)
    f = frame(1)

    assert queue.offer("cam04", f) is None
    cam_id, got, offered_at = queue.take()

    assert (cam_id, offered_at) == ("cam04", 1234.5)
    assert got is f


def test_frames_come_out_in_the_order_they_went_in_whatever_the_camera():
    queue = ac.FrameQueue(5)
    sent = [("cam06", frame(1)), ("cam04", frame(2)),
            ("cam06", frame(3)), ("cam05", frame(4))]
    for cam, f in sent:
        queue.offer(cam, f)

    got = [queue.take() for _ in sent]

    assert [g[0] for g in got] == ["cam06", "cam04", "cam06", "cam05"]
    assert all(g[1] is f for g, (_, f) in zip(got, sent))


def test_a_queue_with_room_drops_nothing():
    queue = ac.FrameQueue(2)

    assert queue.offer("cam04", frame()) is None
    assert queue.offer("cam05", frame()) is None


def test_a_full_queue_drops_its_oldest_frame_to_make_room_for_the_new_one():
    queue = ac.FrameQueue(2)
    oldest, middle, newest = frame(1), frame(2), frame(3)
    queue.offer("cam04", oldest)
    queue.offer("cam04", middle)

    assert queue.offer("cam04", newest) == "cam04"

    assert len(waiting(queue)) == 2
    assert queue.take()[1] is middle
    assert queue.take()[1] is newest


def test_the_drop_is_reported_against_the_camera_the_frame_came_from():
    # cam06's frame pushes cam04's older one out: cam04 lost it.
    queue = ac.FrameQueue(2)
    queue.offer("cam04", frame(1))
    queue.offer("cam05", frame(2))

    assert queue.offer("cam06", frame(3)) == "cam04"

    assert [queue.take()[0] for _ in range(2)] == ["cam05", "cam06"]


def test_all_cameras_share_the_room():
    queue = ac.FrameQueue(3)
    for cam in ("cam04", "cam05", "cam06"):
        assert queue.offer(cam, frame()) is None

    # The queue is full whoever filled it.
    assert queue.offer("cam05", frame()) == "cam04"


def test_taking_a_frame_makes_room_again():
    queue = ac.FrameQueue(1)
    queue.offer("cam04", frame())
    queue.take()

    assert queue.offer("cam04", frame()) is None


def test_a_queue_of_one_always_holds_the_newest_frame():
    queue = ac.FrameQueue(1)
    newest = frame(9)
    for i in range(5):
        queue.offer("cam04", frame(i))
    queue.offer("cam04", newest)

    assert queue.take()[1] is newest


def test_take_blocks_until_a_frame_is_offered():
    queue = ac.FrameQueue(3)
    got = []
    thread = threading.Thread(target=lambda: got.append(queue.take()),
                              daemon=True)
    thread.start()
    thread.join(timeout=0.2)
    assert thread.is_alive()        # still waiting, nothing offered yet

    queue.offer("cam04", frame())
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert got[0][0] == "cam04"


def test_every_offered_frame_is_taken_dropped_or_still_waiting():
    # Three cameras offering as fast as they can against one consumer, as
    # in production. Nothing may be lost or duplicated.
    queue = ac.FrameQueue(5)
    per_camera, cams = 500, ("cam04", "cam05", "cam06")
    dropped, taken = [], []             # list.append is thread-safe
    stop = threading.Event()

    def produce(cam):
        for i in range(per_camera):
            gone = queue.offer(cam, frame(i % 250))
            if gone:
                dropped.append(gone)

    def consume():
        while not stop.is_set():
            taken.append(queue.take()[0])

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    producers = [threading.Thread(target=produce, args=(c,)) for c in cams]
    for p in producers:
        p.start()
    for p in producers:
        p.join(timeout=30)
    stop.set()
    # One more offer wakes a consumer that is blocked in take(); it counts
    # like any other.
    gone = queue.offer("cam04", frame())
    if gone:
        dropped.append(gone)
    consumer.join(timeout=5)
    assert not consumer.is_alive()

    offered = per_camera * len(cams) + 1
    assert len(taken) + len(dropped) + len(waiting(queue)) == offered
    assert set(taken) | set(dropped) <= set(cams)
