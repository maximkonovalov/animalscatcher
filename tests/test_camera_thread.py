"""Tests for camera_thread and the RTSP URL / camera-id plumbing behind the
[CAMERA] channels and rtsp_path keys (previously a hardcoded Hikvision-style
URL inside camera_thread), and for what a camera does with the frames it
reads: check them for motion on its own, and hand over only the frame that
shows it."""
import logging
import time

import numpy as np
import pytest

import ac


class _StopThread(BaseException):
    """Derives from BaseException so it escapes camera_thread's
    `except Exception` loop and lets a test end the otherwise endless
    thread."""


def set_camera(monkeypatch, user="u", password="p", ip="10.0.0.9",
               port="554", rtsp_path="/Streaming/Channels/{channel}02"):
    monkeypatch.setattr(ac, "USER", user)
    monkeypatch.setattr(ac, "PASS", password)
    monkeypatch.setattr(ac, "IP", ip)
    monkeypatch.setattr(ac, "PORT", port)
    monkeypatch.setattr(ac, "RTSP_PATH", rtsp_path)


def test_default_layout_builds_the_url_v014_hardcoded(monkeypatch):
    # Guard for every existing deployment: with no channels/rtsp_path in
    # ac.cfg, channel 4 must open exactly the URL earlier versions built.
    assert ac.CAMERA_CHANNELS == [4, 5, 6]
    set_camera(monkeypatch, user="admin", password="secret", ip="192.168.1.2",
               port="8554", rtsp_path=ac.RTSP_PATH)

    assert ac._rtsp_url(4) == (
        "rtsp://admin:secret@192.168.1.2:8554/Streaming/Channels/402")


@pytest.mark.parametrize("template, channel, expected_path", [
    ("/Streaming/Channels/{channel}01", 2, "/Streaming/Channels/201"),
    ("/cam/realmonitor?channel={channel}&subtype=1", 3,
     "/cam/realmonitor?channel=3&subtype=1"),
    ("/live_{channel:02d}_main", 7, "/live_07_main"),
    ("/stream2", 1, "/stream2"),
])
def test_rtsp_path_template_is_applied(monkeypatch, template, channel,
                                       expected_path):
    set_camera(monkeypatch, rtsp_path=template)

    assert ac._rtsp_url(channel) == f"rtsp://u:p@10.0.0.9:554{expected_path}"


def test_credentials_are_percent_encoded(monkeypatch):
    # A raw @ : / # in either would otherwise be parsed as URL structure.
    set_camera(monkeypatch, user="us@er", password="p:a/s#s")

    url = ac._rtsp_url(1)

    assert url.startswith("rtsp://us%40er:p%3Aa%2Fs%23s@10.0.0.9:554/")


@pytest.mark.parametrize("channel, expected", [
    (1, "cam01"),
    # The default channels must keep their original folder names.
    (4, "cam04"), (5, "cam05"), (6, "cam06"),
    (10, "cam10"),
    (12, "cam12"),
])
def test_cam_id_is_zero_padded_to_two_digits(channel, expected):
    assert ac._cam_id(channel) == expected


# --- what a camera does with its frames -----------------------------------

CAM = 12
CAM_ID = "cam12"


class ScriptedClock:
    """Stands in for the `time` module as ac sees it: time moves only when
    a frame is grabbed, so a test says how far apart the frames are instead
    of waiting for it. Everything else, and logging's own clock, is real."""

    def __init__(self, start=1000.0):
        self.now = start

    def time(self):
        return self.now

    def sleep(self, seconds):
        pass

    def __getattr__(self, name):
        return getattr(time, name)


class ScriptedCapture:
    """A capture that hands out the scripted frames, one grab() each, and
    ends the thread when they run out. A None entry is a failed grab -- the
    stream dropping. Each grab() moves `clock` on by `spacing` seconds."""

    def __init__(self, frames, clock, spacing=0.25):
        self._script = list(frames)
        self._current = None
        self._clock = clock
        self._spacing = spacing
        self.retrieved = 0

    def grab(self):
        if not self._script:
            raise _StopThread
        self._current = self._script.pop(0)
        self._clock.now += self._spacing
        return self._current is not None

    def retrieve(self):
        self.retrieved += 1
        return True, self._current

    def release(self):
        pass


@pytest.fixture
def camera(monkeypatch, tmp_path):
    """Runs camera_thread for channel 12 on a scripted capture and returns
    (capture, the FrameQueue it fed): a new one `size` frames deep, or the
    `queue` a test hands in, say one other cameras already filled. Motion
    is checked on every frame against the previous one (lookback 0, no
    blur) unless a test says otherwise, and the frames arrive `spacing` (a
    quarter of a second) apart."""
    clock = ScriptedClock()
    monkeypatch.setattr(ac, "BASE_OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(ac, "time", clock)
    for name, value in dict(THRESHOLD=0.02, PIXEL_DELTA=25, CHECK_INTERVAL=1,
                            LOOKBACK=0.0, BLUR=1).items():
        monkeypatch.setattr(ac, f"MOTION_{name}", value)
    set_camera(monkeypatch)
    monkeypatch.setitem(ac.stats, "motion", {})
    monkeypatch.setitem(ac.stats, "streams", {})

    def run(frames, spacing=0.25, size=ac.AI_QUEUE_SIZE, queue=None,
            **tuning):
        for name, value in tuning.items():
            monkeypatch.setattr(ac, f"MOTION_{name.upper()}", value)
        if queue is None:
            queue = ac.FrameQueue(size)
        monkeypatch.setattr(ac, "detection_queue", queue)
        cap = ScriptedCapture(frames, clock, spacing)
        monkeypatch.setattr(ac.cv2, "VideoCapture", lambda url, backend: cap)
        with pytest.raises(_StopThread):
            ac.camera_thread(CAM)
        return cap, queue

    return run


def queued(queue):
    """Every frame still waiting in `queue`, oldest first."""
    return [f for _, f, _ in queue._frames]


def quiet():
    return np.full((120, 160, 3), 100, dtype="uint8")


def visitor():
    frame = quiet()
    frame[30:90, 40:120] = 250      # 25% of the frame
    return frame


def test_camera_thread_opens_the_configured_url_and_names_by_channel(
        monkeypatch, tmp_path):
    opened = []

    class FakeCapture:
        def __init__(self, url, backend):
            opened.append(url)

        def grab(self):
            raise _StopThread

    monkeypatch.setattr(ac.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(ac, "BASE_OUTPUT_FOLDER", str(tmp_path))
    set_camera(monkeypatch, rtsp_path="/live/{channel}/sub")

    with pytest.raises(_StopThread):
        ac.camera_thread(12)

    assert opened == ["rtsp://u:p@10.0.0.9:554/live/12/sub"]
    assert (tmp_path / "cam12").is_dir()


def test_a_still_scene_never_reaches_the_ai(camera):
    _, queue = camera([quiet() for _ in range(10)])

    assert queued(queue) == []
    counts = ac.stats["motion"][CAM_ID]
    assert (counts["checks"], counts["motion"]) == (10, 0)


def test_the_frame_that_shows_motion_is_the_one_the_ai_gets(camera):
    # Of the two frames compared, the second -- the one that has the
    # visitor in it -- is handed over.
    shows_visitor = visitor()
    _, queue = camera([quiet(), quiet(), shows_visitor])

    cam_id, frame, _ = queue.take()

    assert cam_id == CAM_ID
    assert frame is shows_visitor
    counts = ac.stats["motion"][CAM_ID]
    assert (counts["checks"], counts["motion"]) == (3, 1)


def test_only_every_nth_frame_is_converted_and_checked(camera):
    cap, _ = camera([quiet() for _ in range(12)], check_interval=4)

    assert cap.retrieved == 3
    assert ac.stats["motion"][CAM_ID]["checks"] == 3


def test_motion_is_found_between_the_frames_that_are_checked(camera):
    # Frames 4 and 8 are the checked ones; the visitor is in frame 8 only,
    # and frames in between are never looked at.
    frames = [quiet() for _ in range(8)]
    frames[7] = visitor()
    _, queue = camera(frames, check_interval=4)

    assert queue.take()[1] is frames[7]


def test_frames_with_motion_queue_up_for_the_ai_oldest_first(camera):
    # The visitor appears (motion), then is gone again (motion again)
    # before anyone took the first frame: both wait for the AI.
    appears, gone = visitor(), quiet()
    _, queue = camera([quiet(), appears, gone])

    assert queue.take()[1] is appears
    assert queue.take()[1] is gone
    counts = ac.stats["motion"][CAM_ID]
    assert (counts["motion"], counts["dropped"]) == (2, 0)


def test_a_full_queue_drops_its_oldest_frames_and_counts_them(camera):
    # Every check after the first sees the visitor come or go: five frames
    # with motion for a queue two deep. The newest two are kept; the three
    # before them were dropped, and each is counted as motion and as a drop.
    frames = [quiet()] + [visitor() if i % 2 == 0 else quiet()
                          for i in range(5)]
    _, queue = camera(frames, size=2)

    assert len(queued(queue)) == 2
    assert queue.take()[1] is frames[4]
    assert queue.take()[1] is frames[5]
    counts = ac.stats["motion"][CAM_ID]
    assert (counts["motion"], counts["dropped"]) == (5, 3)


def test_a_drop_is_counted_against_the_camera_the_frame_came_from(camera):
    # The queue is full of other cameras' frames. This camera's motion frame
    # pushes the oldest, cam04's, out: cam04 lost a frame, this camera
    # didn't.
    queue = ac.FrameQueue(2)
    queue.offer("cam04", quiet())
    queue.offer("cam05", quiet())

    camera([quiet(), visitor()], queue=queue)

    assert ac.stats["motion"]["cam04"]["dropped"] == 1
    assert "cam05" not in ac.stats["motion"]
    assert ac.stats["motion"][CAM_ID]["dropped"] == 0
    assert [queue.take()[0] for _ in range(2)] == ["cam05", CAM_ID]


def test_a_drop_does_not_stop_the_camera_or_its_motion_log(camera, log_lines):
    # The queue stays full for the whole burst; the camera keeps checking,
    # and the burst is still one log line.
    frames = [quiet()] + [visitor() if i % 2 == 0 else quiet()
                          for i in range(8)]
    camera(frames, size=1)

    counts = ac.stats["motion"][CAM_ID]
    assert (counts["checks"], counts["motion"], counts["dropped"]) == (9, 8, 7)
    assert len(log_lines) == 1


def test_the_peak_change_is_recorded_whether_or_not_it_triggered(camera):
    faint = quiet()
    faint[0:30, 0:40] = 250             # 6.25% of the frame
    camera([quiet(), faint], threshold=0.5)

    counts = ac.stats["motion"][CAM_ID]
    assert counts["motion"] == 0
    assert counts["peak"] == pytest.approx(30 * 40 / (120 * 160), abs=0.005)


def test_a_dropped_stream_is_marked_offline_and_comparison_starts_afresh(
        camera):
    # The first frame after the stream comes back has nothing before the
    # outage to be compared with, however different it is.
    _, queue = camera([quiet(), None, visitor()])

    assert queued(queue) == []
    counts = ac.stats["motion"][CAM_ID]
    assert (counts["checks"], counts["motion"]) == (2, 0)


def test_the_stream_is_reported_online_with_its_resolution(camera):
    camera([quiet(), quiet()])

    assert ac.stats["streams"][CAM_ID] == {"status": "ONLINE",
                                           "res": "160x120"}


def test_the_configured_threshold_decides_what_counts_as_motion(camera):
    # A visitor covering a quarter of the frame is motion under a 2%
    # threshold and not under a 50% one.
    camera([quiet(), visitor()], threshold=0.5)
    assert ac.stats["motion"][CAM_ID]["motion"] == 0

    ac.stats["motion"].clear()
    camera([quiet(), visitor()], threshold=0.02)
    assert ac.stats["motion"][CAM_ID]["motion"] == 1


# --- the motion log ---------------------------------------------------------

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


def still(frame, seconds, spacing=0.25):
    """Copies of `frame` covering `seconds` of the stream."""
    return [frame] * round(seconds / spacing)


def test_a_still_scene_logs_no_motion(camera, log_lines):
    camera([quiet() for _ in range(10)])

    assert log_lines == []


def test_motion_is_logged_with_its_camera_size_and_threshold(
        camera, log_lines):
    camera([quiet(), visitor()])

    # The visitor covers 25% of the frame; the threshold is the fixture's 2%.
    assert log_lines == [
        "[MOTION] cam12: 25.000% of the frame changed (threshold 2.000%)"]


def test_only_the_first_motion_of_a_burst_is_logged(camera, log_lines):
    # The visitor comes and goes: every check after the first sees a change
    # too, and each is counted, but the log gets one line.
    camera([quiet(), visitor(), quiet(), visitor(), quiet()])

    assert ac.stats["motion"][CAM_ID]["motion"] == 4
    assert len(log_lines) == 1


def test_motion_after_a_quiet_spell_is_a_new_event(camera, log_lines):
    # Twelve seconds without a change -- longer than MOTION_LOG_GAP --
    # and then the visitor is gone.
    shows_visitor = visitor()
    camera([quiet(), shows_visitor] + still(shows_visitor, 12) + [quiet()])

    assert ac.stats["motion"][CAM_ID]["motion"] == 2
    assert len(log_lines) == 2


def test_motion_within_the_gap_is_still_the_same_event(camera, log_lines):
    shows_visitor = visitor()
    camera([quiet(), shows_visitor] + still(shows_visitor, 5) + [quiet()])

    assert ac.stats["motion"][CAM_ID]["motion"] == 2
    assert len(log_lines) == 1


def test_an_event_lasts_as_long_as_motion_keeps_coming(camera, log_lines):
    # A change every 6 seconds for 18: each is within the gap of the one
    # before, though the last is far from the first -- the gap runs from
    # the latest motion, not from where the event began.
    gone, here = quiet(), visitor()
    camera([gone, here] + still(here, 5.75) + [gone] + still(gone, 5.75)
           + [here] + still(here, 5.75) + [gone])

    assert ac.stats["motion"][CAM_ID]["motion"] == 4
    assert len(log_lines) == 1
