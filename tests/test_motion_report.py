"""Tests for ac._motion_report -- the per-camera MOTION block of the
Telegram summary, which is how a threshold gets tuned from real data."""
import ac


def counters(**overrides):
    base = {"checks": 0, "motion": 0, "analysed": 0, "animals": 0,
            "replaced": 0, "peak": 0.0, "wait_sum": 0.0, "wait_max": 0.0}
    return {**base, **overrides}


def test_every_streamed_camera_is_listed_even_before_it_counted_anything():
    report = ac._motion_report({})

    for cam_id in ("cam04", "cam05", "cam06"):
        assert (f"- {cam_id}: 0 checks, 0 motion, 0 analysed "
                "(0 with an animal), 0 replaced by a newer frame") in report
    # Nothing was analysed, so there is no wait to average.
    assert "AI wait" not in report


def test_counts_peak_threshold_and_ai_wait_are_reported():
    report = ac._motion_report({"cam05": counters(
        checks=345600, motion=52, analysed=48, animals=6, replaced=4,
        peak=0.0031, wait_sum=100.8, wait_max=9.8)})

    assert ("- cam05: 345600 checks, 52 motion, 48 analysed "
            "(6 with an animal), 4 replaced by a newer frame") in report
    assert "strongest change 0.310% (threshold 0.120%)" in report
    assert "AI wait avg 2.1s, max 9.8s" in report


def test_each_camera_reports_its_own_animals():
    report = ac._motion_report({
        "cam04": counters(analysed=10, animals=3),
        "cam06": counters(analysed=40, animals=0)})

    assert "- cam04: 0 checks, 0 motion, 10 analysed (3 with an animal)" in report
    assert "- cam05: 0 checks, 0 motion, 0 analysed (0 with an animal)" in report
    assert "- cam06: 0 checks, 0 motion, 40 analysed (0 with an animal)" in report


def test_every_camera_is_shown_against_the_configured_threshold(monkeypatch):
    monkeypatch.setattr(ac, "MOTION_THRESHOLD", 0.001)

    report = ac._motion_report({})

    assert report.count("threshold 0.100%") == 3
