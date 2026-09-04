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
import os

import numpy as np
import pytest

import ac

NAMES = {0: "Animal", 1: "Person", 2: "Vehicle"}
COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}


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


def test_unmapped_detector_class_does_not_raise(frame, capture_photo_submits):
    """Regression test for a fixed KeyError: stats[obj_name] += 1 used to
    crash for any class_id outside {0,1,2} (obj_name falls back to the
    literal "Object", a key `stats` never had).

    In today's code this path is actually gated by `motion_val`, which
    _process_frame only ever populates for classes 0-2 (see the
    `for c in [0, 1, 2]` loop at the end of the function) -- so a
    class_id outside that set can never reach the alert branch through
    the normal call sequence. We seed `motion_val` directly here to
    exercise the fixed line itself, as a guard against a future change
    (e.g. tracking additional classes) making this reachable again.
    """
    classifier = FakeClassifier()
    cam_id = "cam04"
    d_key = (cam_id, 9)
    last_det, motion_val, last_box = {}, {d_key: True}, {}
    detector = FakeDetector([0.9], [9], [[10, 10, 50, 50]])

    ac._process_frame(cam_id, frame, detector, classifier, NAMES, COLORS,
                      last_det, motion_val, last_box)  # would previously raise

    assert ac.stats["Object"] == 1
    del ac.stats["Object"]


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
