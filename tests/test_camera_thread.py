"""Tests for the RTSP URL / camera-id plumbing behind the [CAMERA]
channels and rtsp_path keys -- previously a hardcoded Hikvision-style URL
inside camera_thread."""
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


def test_camera_thread_opens_the_configured_url_and_names_by_channel(
        monkeypatch, tmp_path):
    opened = []

    class FakeCapture:
        def __init__(self, url, backend):
            opened.append(url)

        def read(self):
            raise _StopThread

    monkeypatch.setattr(ac.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(ac, "BASE_OUTPUT_FOLDER", str(tmp_path))
    set_camera(monkeypatch, rtsp_path="/live/{channel}/sub")

    with pytest.raises(_StopThread):
        ac.camera_thread(12)

    assert opened == ["rtsp://u:p@10.0.0.9:554/live/12/sub"]
    assert (tmp_path / "cam12").is_dir()
