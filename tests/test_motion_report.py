"""Tests for the summary's motion and queue reporting: the per-camera MOTION
block (_motion_report), which is how a threshold gets tuned from real data,
the AI QUEUE block (_queue_report), which says how many frames the queue
dropped, and the summary the two end up in."""
import datetime

import pytest

import ac


def counters(**overrides):
    base = {"checks": 0, "motion": 0, "analysed": 0, "animals": 0,
            "dropped": 0, "peak": 0.0, "wait_sum": 0.0, "wait_max": 0.0}
    return {**base, **overrides}


def test_every_streamed_camera_is_listed_even_before_it_counted_anything():
    report = ac._motion_report({})

    for cam_id in ("cam04", "cam05", "cam06"):
        assert (f"- {cam_id}: 0 checks, 0 motion, 0 analysed "
                "(0 with an animal), 0 dropped (queue full)") in report
    # Nothing was analysed, so there is no wait to average.
    assert "AI wait" not in report


def test_counts_peak_threshold_and_ai_wait_are_reported():
    report = ac._motion_report({"cam05": counters(
        checks=345600, motion=52, analysed=48, animals=6, dropped=4,
        peak=0.0031, wait_sum=100.8, wait_max=9.8)})

    assert ("- cam05: 345600 checks, 52 motion, 48 analysed "
            "(6 with an animal), 4 dropped (queue full)") in report
    assert "strongest change 0.310% (threshold 0.120%)" in report
    assert "AI wait avg 2.1s, max 9.8s" in report


def test_each_camera_reports_its_own_animals():
    report = ac._motion_report({
        "cam04": counters(analysed=10, animals=3),
        "cam06": counters(analysed=40, animals=0)})

    assert "- cam04: 0 checks, 0 motion, 10 analysed (3 with an animal)" in report
    assert "- cam05: 0 checks, 0 motion, 0 analysed (0 with an animal)" in report
    assert "- cam06: 0 checks, 0 motion, 40 analysed (0 with an animal)" in report


def test_each_camera_reports_its_own_drops():
    report = ac._motion_report({
        "cam04": counters(motion=5, analysed=5, dropped=0),
        "cam06": counters(motion=90, analysed=6, dropped=70)})

    assert "5 analysed (0 with an animal), 0 dropped (queue full)" in report
    assert "6 analysed (0 with an animal), 70 dropped (queue full)" in report


def test_every_camera_is_shown_against_the_configured_threshold(monkeypatch):
    monkeypatch.setattr(ac, "MOTION_THRESHOLD", 0.001)

    report = ac._motion_report({})

    assert report.count("threshold 0.100%") == 3


def test_the_queue_report_adds_up_every_camera():
    report = ac._queue_report({
        "cam04": counters(motion=40, dropped=10),
        "cam05": counters(),
        "cam06": counters(motion=60, dropped=15)})

    assert report == ("- Offered: 100 (frames that showed motion)\n"
                      "- Dropped (queue full): 25 (25.0%)")


def test_the_queue_report_with_nothing_offered_is_zero_not_a_division_error():
    assert ac._queue_report({}) == (
        "- Offered: 0 (frames that showed motion)\n"
        "- Dropped (queue full): 0 (0.0%)")


def test_a_small_share_of_drops_keeps_its_decimal():
    report = ac._queue_report({"cam04": counters(motion=300, dropped=1)})

    assert "- Dropped (queue full): 1 (0.3%)" in report


class _StopSummary(BaseException):
    """Derives from BaseException so it escapes summary_engine's
    `except Exception` and ends the otherwise endless loop."""


def test_the_summary_carries_the_motion_and_queue_blocks_then_starts_over(
        monkeypatch):
    sent, sleeps = [], []

    def sleep(seconds):
        # The first sleep is the wait for the summary; the second ends the
        # test, after the one report.
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise _StopSummary

    stats = {"Animal": 3, "Person": 1,
             "start_time": datetime.datetime(2026, 10, 31, 21, 22),
             "streams": {"cam06": {"status": "ONLINE", "res": "1280x720"}},
             "motion": {"cam06": counters(
                 checks=900, motion=40, analysed=12, dropped=28,
                 peak=0.004, wait_sum=24.0, wait_max=7.0)}}
    monkeypatch.setattr(ac, "stats", stats)
    monkeypatch.setattr(ac.time, "sleep", sleep)
    monkeypatch.setattr(ac, "send_telegram_message", sent.append)

    with pytest.raises(_StopSummary):
        ac.summary_engine()

    (report,) = sent
    assert "Range: 31/10/2026 21:22 - " in report
    assert "DETECTIONS:\n- Animals: 3\n- People: 1\n\nMOTION:" in report
    assert "Vehicle" not in report
    assert ("- cam06: 900 checks, 40 motion, 12 analysed (0 with an animal), "
            "28 dropped (queue full)") in report
    assert (f"AI QUEUE (room for {ac.AI_QUEUE_SIZE} frames, all cameras):\n"
            "- Offered: 40 (frames that showed motion)\n"
            "- Dropped (queue full): 28 (70.0%)") in report
    assert report.index("MOTION:") < report.index("AI QUEUE")
    # The next summary counts from zero.
    assert stats["motion"] == {}
    assert (stats["Animal"], stats["Person"]) == (0, 0)
