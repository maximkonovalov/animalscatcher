"""Tests for ac._process_frame's detection/debounce/static-filter/alert
logic, using a fake detector and classifier so no real AI models or
network calls are involved.

Debounce mechanics (verified against the implementation, not assumed):
  - A (camera, class) only enters the alert/static check once
    `motion_val` is True for it, which only happens starting on the
    *second* consecutive processed frame containing that class (the
    first frame just registers motion for next time).
  - `last_box` is only populated when an alert actually fires. So the
    very first alert for a (camera, class) always fires on that second
    sighting, regardless of box position -- there's nothing recorded
    yet to compare against.
  - Only a *third* (or later) sighting is compared against the box
    recorded at the last alert, and only if COOLDOWN seconds have
    elapsed since then; if it's within STATIC_TOLERANCE of that box,
    it's filtered as static instead of alerting again.
"""
import logging
import os

import numpy as np
import pytest

import ac

NAMES = {0: "Animal", 1: "Person"}
COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}


class FakeDetections:
    def __init__(self, confidence, class_id, xyxy):
        self.confidence = confidence
        self.class_id = class_id
        self.xyxy = xyxy


class FakeDetector:
    """Returns a fixed set of detections regardless of the input frame."""

    def __init__(self, confidence, class_id, xyxy):
        self._detections = FakeDetections(confidence, class_id, xyxy)

    def single_image_detection(self, frame):
        return {"detections": self._detections}


class FakeClassifier:
    def __init__(self, label="Coyote", confidence=0.8):
        self._label = label
        self._confidence = confidence
        self.last_crop_shape = None

    def single_image_classification(self, crop):
        self.last_crop_shape = crop.shape
        return {"label": self._label, "confidence": self._confidence}


@pytest.fixture
def frame():
    return np.zeros((100, 100, 3), dtype="uint8")


@pytest.fixture(autouse=True)
def snaps_dir(tmp_path, monkeypatch):
    """Point BASE_OUTPUT_FOLDER at a throwaway dir and pre-create the
    cam04 subfolder (normally done by camera_thread's os.makedirs)."""
    monkeypatch.setattr(ac, "BASE_OUTPUT_FOLDER", str(tmp_path))
    os.makedirs(tmp_path / "cam04", exist_ok=True)
    return tmp_path


@pytest.fixture(autouse=True)
def reset_animal_stat():
    ac.stats["Animal"] = 0
    yield
    ac.stats["Animal"] = 0


@pytest.fixture(autouse=True)
def fresh_motion_stats(monkeypatch):
    """The per-camera counters behind the summary's MOTION block."""
    monkeypatch.setitem(ac.stats, "motion", {})


@pytest.fixture
def capture_photo_submits(monkeypatch):
    """Replace photo_executor.submit with a synchronous recorder so
    tests don't spawn real threads or hit the network."""
    calls = []

    def fake_submit(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))

        class FakeFuture:
            def result(self):
                return None
        return FakeFuture()

    monkeypatch.setattr(ac.photo_executor, "submit", fake_submit)
    return calls


def run(detector, classifier, frame, state, cam_id="cam04"):
    last_det, motion_val, last_box = state
    ac._process_frame(cam_id, frame, detector, classifier, NAMES, COLORS,
                      last_det, motion_val, last_box)


def test_first_sighting_registers_motion_but_does_not_alert(
        frame, capture_photo_submits):
    detector = FakeDetector([0.9], [0], [[10, 10, 50, 50]])
    classifier = FakeClassifier()
    state = ({}, {}, {})

    run(detector, classifier, frame, state)

    assert ac.stats["Animal"] == 0
    assert capture_photo_submits == []


def test_second_consecutive_sighting_triggers_first_alert(
        frame, capture_photo_submits):
    detector = FakeDetector([0.9], [0], [[10, 10, 50, 50]])
    classifier = FakeClassifier()
    state = ({}, {}, {})

    run(detector, classifier, frame, state)  # registers motion only
    run(detector, classifier, frame, state)  # -> first alert

    assert ac.stats["Animal"] == 1
    assert len(capture_photo_submits) == 1
    fn, args, kwargs = capture_photo_submits[0]
    assert fn is ac.send_telegram_photo
    fpath, caption = args
    assert fpath.endswith(".jpg")
    assert "cam04_Animal_" in fpath
    assert "ALERT: Animal" in caption


def test_third_sighting_in_same_box_is_filtered_as_static(
        frame, capture_photo_submits, monkeypatch):
    # Remove cooldown as a variable so only the box-position comparison
    # is under test.
    monkeypatch.setattr(ac, "COOLDOWN", 0)
    classifier = FakeClassifier()
    same_box = [[10, 10, 50, 50]]
    state = ({}, {}, {})

    run(FakeDetector([0.9], [0], same_box), classifier, frame, state)
    run(FakeDetector([0.9], [0], same_box), classifier, frame, state)  # alert 1
    run(FakeDetector([0.9], [0], same_box), classifier, frame, state)  # static

    assert ac.stats["Animal"] == 1  # only the first alert counted
    assert len(capture_photo_submits) == 1


def test_third_sighting_with_moved_box_alerts_again(
        frame, capture_photo_submits, monkeypatch):
    monkeypatch.setattr(ac, "COOLDOWN", 0)
    classifier = FakeClassifier()
    state = ({}, {}, {})

    run(FakeDetector([0.9], [0], [[10, 10, 50, 50]]), classifier, frame,
        state)
    run(FakeDetector([0.9], [0], [[10, 10, 50, 50]]), classifier, frame,
        state)  # alert 1, box (10,10) recorded
    # Well beyond STATIC_TOLERANCE (3% of a 100px frame == 3px).
    run(FakeDetector([0.9], [0], [[40, 40, 80, 80]]), classifier, frame,
        state)  # moved -> alert 2

    assert ac.stats["Animal"] == 2
    assert len(capture_photo_submits) == 2


def test_cooldown_suppresses_a_third_sighting_even_if_moved(
        frame, capture_photo_submits, monkeypatch):
    monkeypatch.setattr(ac, "COOLDOWN", 9999)  # effectively "never expires"
    classifier = FakeClassifier()
    state = ({}, {}, {})

    run(FakeDetector([0.9], [0], [[10, 10, 50, 50]]), classifier, frame,
        state)
    run(FakeDetector([0.9], [0], [[10, 10, 50, 50]]), classifier, frame,
        state)  # alert 1
    run(FakeDetector([0.9], [0], [[70, 70, 90, 90]]), classifier, frame,
        state)  # moved, but within cooldown window -> suppressed

    assert ac.stats["Animal"] == 1
    assert len(capture_photo_submits) == 1


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


@pytest.mark.parametrize("class_id", [2, 9])
def test_a_class_that_is_not_detected_is_ignored_completely(
        frame, capture_photo_submits, log_lines, class_id):
    # Vehicles are MegaDetector's class 2; 9 stands for anything else it
    # might ever report (this used to be a KeyError on stats["Object"]).
    # Neither leaves a trace: no log line, no registered motion, no count,
    # and no alert however many times it is seen.
    detector = FakeDetector([0.99], [class_id], [[10, 10, 50, 50]])
    state = ({}, {}, {})

    for _ in range(3):
        run(detector, FakeClassifier(), frame, state)

    assert log_lines == []
    assert capture_photo_submits == []
    assert (ac.stats["Animal"], ac.stats["Person"]) == (0, 0)
    assert "Vehicle" not in ac.stats and "Object" not in ac.stats
    assert state[1] == {("cam04", 0): False, ("cam04", 1): False}


def test_a_vehicle_does_not_hide_an_animal_in_the_same_frame(
        frame, capture_photo_submits, log_lines):
    detector = FakeDetector([0.99, 0.9], [2, 0],
                            [[60, 60, 95, 95], [10, 10, 50, 50]])
    state = ({}, {}, {})

    run(detector, FakeClassifier(), frame, state)
    run(detector, FakeClassifier(), frame, state)

    # Only the animal is logged, and it alerts on the second look as usual.
    assert log_lines == ["[DETECT] Animal conf=0.90 on cam04"] * 2
    assert ac.stats["Animal"] == 1
    assert len(capture_photo_submits) == 1


def test_no_detections_leaves_stats_unchanged(frame):
    detector = FakeDetector([], [], [])
    classifier = FakeClassifier()
    state = ({}, {}, {})

    run(detector, classifier, frame, state)

    assert ac.stats["Animal"] == 0


def test_species_classification_crop_is_padded_beyond_raw_box(
        frame, capture_photo_submits):
    # Raw box is 40x40 ([10,10,50,50]); CROP_PADDING (default 0.15)
    # should make the actual classification crop strictly larger on
    # each side, so a tight box doesn't clip a tail/ear/antler.
    detector = FakeDetector([0.9], [0], [[10, 10, 50, 50]])
    classifier = FakeClassifier()
    state = ({}, {}, {})

    run(detector, classifier, frame, state)

    assert classifier.last_crop_shape is not None
    height, width = classifier.last_crop_shape[:2]
    assert height > 40
    assert width > 40


def test_species_classification_crop_padding_clamps_to_frame_bounds(
        frame, capture_photo_submits):
    # Box touching the frame's top-left corner: padding would push
    # coordinates negative before clamping.
    detector = FakeDetector([0.9], [0], [[0, 0, 20, 20]])
    classifier = FakeClassifier()
    state = ({}, {}, {})

    run(detector, classifier, frame, state)

    assert classifier.last_crop_shape is not None
    height, width = classifier.last_crop_shape[:2]
    assert 0 < height <= frame.shape[0]
    assert 0 < width <= frame.shape[1]


# --- the two-frame latch when frames only follow motion --------------------

def test_a_recent_latch_is_kept(monkeypatch):
    monkeypatch.setattr(ac, "CONFIRM_WINDOW", 30)
    motion_val = {("cam04", 0): True}
    last_look = {"cam04": 100.0}

    ac._expire_stale_latch(motion_val, last_look, "cam04", 125.0)

    assert motion_val[("cam04", 0)] is True
    assert last_look["cam04"] == 125.0


def test_an_old_latch_is_forgotten_for_that_camera_only(monkeypatch):
    monkeypatch.setattr(ac, "CONFIRM_WINDOW", 30)
    motion_val = {("cam04", 0): True, ("cam04", 1): True,
                  ("cam05", 0): True}
    last_look = {"cam04": 100.0, "cam05": 100.0}

    ac._expire_stale_latch(motion_val, last_look, "cam04", 131.0)

    assert motion_val[("cam04", 0)] is False
    assert motion_val[("cam04", 1)] is False
    assert motion_val[("cam05", 0)] is True        # another camera's latch
    assert last_look["cam04"] == 131.0


def test_the_first_look_at_a_camera_clears_nothing(monkeypatch):
    monkeypatch.setattr(ac, "CONFIRM_WINDOW", 30)
    motion_val = {}
    last_look = {}

    ac._expire_stale_latch(motion_val, last_look, "cam04", 5000.0)

    assert motion_val == {}
    assert last_look == {"cam04": 5000.0}


def test_a_sighting_is_not_confirmed_by_one_from_long_before(
        frame, capture_photo_submits, monkeypatch):
    # Frames used to arrive around the clock; now only motion brings one,
    # so "the previous frame" can be an hour old. It must not count.
    monkeypatch.setattr(ac, "CONFIRM_WINDOW", 30)
    detector = FakeDetector([0.9], [0], [[10, 10, 50, 50]])
    classifier = FakeClassifier()
    state = ({}, {}, {})
    last_look = {}

    def look(at):
        ac._expire_stale_latch(state[1], last_look, "cam04", at)
        run(detector, classifier, frame, state)

    look(0)             # a sighting, awaiting its confirmation
    look(3600)          # an hour later: a first sighting again
    assert ac.stats["Animal"] == 0

    look(3610)          # ...confirmed by the look ten seconds before
    assert ac.stats["Animal"] == 1


# --- animals counted for the summary's MOTION block -------------------------

def animals_counted(cam_id="cam04"):
    return ac.stats["motion"].get(cam_id, {}).get("animals", 0)


def test_a_frame_with_an_animal_is_counted_even_when_it_does_not_alert(
        frame, capture_photo_submits):
    # A first sighting only registers motion for the next look -- no alert
    # -- but the motion frame did have an animal on it.
    run(FakeDetector([0.9], [0], [[10, 10, 50, 50]]), FakeClassifier(),
        frame, ({}, {}, {}))

    assert animals_counted() == 1
    assert ac.stats["Animal"] == 0


def test_a_frame_is_counted_once_however_many_animals_are_on_it(
        frame, capture_photo_submits):
    detector = FakeDetector([0.9, 0.8], [0, 0],
                            [[10, 10, 30, 30], [50, 50, 90, 90]])

    run(detector, FakeClassifier(), frame, ({}, {}, {}))

    assert animals_counted() == 1


@pytest.mark.parametrize("confidence, class_id", [
    (0.3, 0),       # an animal, but below its threshold
    (0.99, 1),      # a person
    (0.99, 2),      # a vehicle, which is not detected at all
])
def test_a_frame_without_a_confident_animal_is_not_counted(
        frame, capture_photo_submits, confidence, class_id):
    detector = FakeDetector([confidence], [class_id], [[10, 10, 50, 50]])

    run(detector, FakeClassifier(), frame, ({}, {}, {}))

    assert animals_counted() == 0


def test_animals_are_counted_per_camera(frame, capture_photo_submits):
    animal = FakeDetector([0.9], [0], [[10, 10, 50, 50]])
    nothing = FakeDetector([], [], [])

    for cam_id in ("cam04", "cam05", "cam05"):
        run(animal, FakeClassifier(), frame, ({}, {}, {}), cam_id)
    run(nothing, FakeClassifier(), frame, ({}, {}, {}), "cam06")

    assert [animals_counted(c) for c in ("cam04", "cam05", "cam06")] == [
        1, 2, 0]
