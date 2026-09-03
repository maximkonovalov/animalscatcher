"""Tests for ac.load_config -- the config file parsing/validation added
in v0.6 after an ac.cfg missing a required key used to crash with a raw,
unhandled traceback and no log record."""
import configparser

import pytest

import ac

VALID_CFG = """\
[CAMERA]
user = camuser
pass = campass
ip = 192.168.1.50
port = 8554

[TELEGRAM]
token = 123456:ABCDEF
chat_id = -100123456

[DETECTION]
threshold_0 = 0.45
threshold_1 = 0.75
threshold_2 = 0.95
cooldown = 20
frame_interval = 10
summary_interval = 6
species_threshold = 0.45
static_tolerance = 0.03

[PATHS]
base_output_folder = /tmp/snaps
log_file = /tmp/ac_log.txt

[CLEANUP]
max_age_days = 7
cleanup_interval = 24
max_log_size_mb = 5
"""


def write_cfg(tmp_path, content, name="ac.cfg"):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_valid_config_returns_expected_values(tmp_path):
    path = write_cfg(tmp_path, VALID_CFG)
    cfg = ac.load_config(path)

    assert cfg["user"] == "camuser"
    assert cfg["pass"] == "campass"
    assert cfg["ip"] == "192.168.1.50"
    assert cfg["port"] == "8554"
    assert cfg["telegram_token"] == "123456:ABCDEF"
    assert cfg["telegram_chat_id"] == "-100123456"
    assert cfg["thresholds"] == {0: 0.45, 1: 0.75, 2: 0.95}
    assert cfg["cooldown"] == 20
    assert cfg["frame_interval"] == 10
    assert cfg["summary_interval"] == 6
    assert cfg["species_threshold"] == 0.45
    assert cfg["static_tolerance"] == 0.03
    assert cfg["max_age_days"] == 7
    assert cfg["cleanup_interval"] == 24
    assert cfg["max_log_size_mb"] == 5


def test_missing_file_raises_systemexit_with_path(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.cfg")

    with pytest.raises(SystemExit) as exc_info:
        ac.load_config(missing_path)

    assert missing_path in str(exc_info.value)
    assert "not found" in str(exc_info.value).lower()


def test_empty_file_raises_systemexit(tmp_path):
    # A file that exists but has no sections at all.
    path = write_cfg(tmp_path, "")

    with pytest.raises(SystemExit) as exc_info:
        ac.load_config(path)

    assert "ac.cfg.example" in str(exc_info.value)


def test_missing_key_raises_friendly_systemexit(tmp_path):
    # Drop species_threshold, the key that was added in v0.5/v0.6 and has
    # already caused a real "missing key" incident for this project.
    cfg_without_species_threshold = VALID_CFG.replace(
        "species_threshold = 0.45\n", "")
    path = write_cfg(tmp_path, cfg_without_species_threshold)

    with pytest.raises(SystemExit) as exc_info:
        ac.load_config(path)

    message = str(exc_info.value)
    assert "species_threshold" in message
    assert "ac.cfg.example" in message


def test_missing_section_raises_friendly_systemexit(tmp_path):
    # Drop the entire [CLEANUP] section.
    lines = VALID_CFG.splitlines()
    cleanup_start = lines.index("[CLEANUP]")
    cfg_without_cleanup = "\n".join(lines[:cleanup_start])
    path = write_cfg(tmp_path, cfg_without_cleanup)

    with pytest.raises(SystemExit) as exc_info:
        ac.load_config(path)

    assert "CLEANUP" in str(exc_info.value)


def test_invalid_numeric_value_raises_friendly_systemexit(tmp_path):
    # cooldown must be an int; a non-numeric value used to raise an
    # uncaught ValueError instead of the friendly SystemExit that
    # getfloat/getint failures for missing *keys* already got.
    cfg_with_bad_cooldown = VALID_CFG.replace(
        "cooldown = 20", "cooldown = not-a-number")
    path = write_cfg(tmp_path, cfg_with_bad_cooldown)

    with pytest.raises(SystemExit) as exc_info:
        ac.load_config(path)

    assert "ac.cfg.example" in str(exc_info.value)


def test_unreadable_or_missing_file_error_is_not_a_bare_traceback(tmp_path):
    # Sanity check that load_config never lets a raw configparser.Error
    # or ValueError escape uncaught -- always SystemExit.
    path = write_cfg(tmp_path, VALID_CFG.replace(
        "threshold_0 = 0.45", "threshold_0 = nope"))

    with pytest.raises(SystemExit):
        ac.load_config(path)
    # Confirm it specifically isn't a configparser.Error/ValueError
    # propagating out uncaught (which pytest.raises(SystemExit) above
    # would already fail to catch if it did -- this documents intent).
    with pytest.raises((SystemExit,)):
        try:
            ac.load_config(path)
        except (configparser.Error, ValueError):
            pytest.fail("load_config leaked a raw config exception "
                       "instead of raising SystemExit")
