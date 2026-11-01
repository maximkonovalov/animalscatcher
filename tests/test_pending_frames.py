"""Tests for ac.PendingFrames -- the hand-off from the camera threads to the
single ai_engine thread: one waiting frame per camera (the newest), served
camera by camera in turn."""
import threading

import numpy as np

import ac


def frame(value=0):
    return np.full((5, 5, 3), value, dtype="uint8")


def test_take_returns_the_offered_frame_with_its_camera_and_offer_time(
        monkeypatch):
    monkeypatch.setattr(ac.time, "time", lambda: 1234.5)
    pending = ac.PendingFrames()
    f = frame(1)

    pending.offer("cam04", f)
    cam_id, got, offered_at = pending.take()

    assert (cam_id, offered_at) == ("cam04", 1234.5)
    assert got is f


def test_a_newer_frame_replaces_one_nobody_picked_up():
    pending = ac.PendingFrames()
    old, new = frame(1), frame(2)

    assert pending.offer("cam04", old) is False
    assert pending.offer("cam04", new) is True

    _, got, _ = pending.take()
    assert got is new
    # Only the newest was kept: nothing else is waiting.
    assert pending._waiting == {}


def test_cameras_are_served_in_turn():
    pending = ac.PendingFrames()
    for cam in ("cam06", "cam04", "cam05"):
        pending.offer(cam, frame())

    assert [pending.take()[0] for _ in range(3)] == ["cam04", "cam05", "cam06"]


def test_a_camera_that_offers_again_goes_to_the_back_of_the_turn():
    # cam04 keeps showing motion, but cam05 and cam06 have waited: they
    # are served before cam04 comes round again.
    pending = ac.PendingFrames()
    for cam in ("cam04", "cam05", "cam06"):
        pending.offer(cam, frame())
    assert pending.take()[0] == "cam04"

    pending.offer("cam04", frame())

    assert [pending.take()[0] for _ in range(3)] == ["cam05", "cam06", "cam04"]


def test_the_turn_wraps_around():
    pending = ac.PendingFrames()
    pending.offer("cam06", frame())
    assert pending.take()[0] == "cam06"

    pending.offer("cam04", frame())

    assert pending.take()[0] == "cam04"


def test_a_lone_camera_is_served_every_time():
    pending = ac.PendingFrames()
    served = []
    for _ in range(3):
        pending.offer("cam05", frame())
        served.append(pending.take()[0])

    assert served == ["cam05"] * 3


def test_take_blocks_until_a_frame_is_offered():
    pending = ac.PendingFrames()
    got = []
    thread = threading.Thread(target=lambda: got.append(pending.take()),
                              daemon=True)
    thread.start()
    thread.join(timeout=0.2)
    assert thread.is_alive()        # still waiting, nothing offered yet

    pending.offer("cam04", frame())
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert got[0][0] == "cam04"


def test_every_offered_frame_is_taken_replaced_or_still_waiting():
    # Three cameras offering as fast as they can against one consumer, as
    # in production. Nothing may be lost or duplicated, and every camera
    # must get served.
    pending = ac.PendingFrames()
    per_camera, cams = 500, ("cam04", "cam05", "cam06")
    replaced = {c: 0 for c in cams}
    taken = []
    stop = threading.Event()

    def produce(cam):
        for i in range(per_camera):
            if pending.offer(cam, frame(i % 250)):
                replaced[cam] += 1

    def consume():
        while not stop.is_set():
            taken.append(pending.take()[0])

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
    if pending.offer("cam04", frame()):
        replaced["cam04"] += 1
    consumer.join(timeout=5)
    assert not consumer.is_alive()

    leftover = len(pending._waiting)
    offered = per_camera * len(cams) + 1
    assert len(taken) + sum(replaced.values()) + leftover == offered
    assert set(taken) == set(cams)
