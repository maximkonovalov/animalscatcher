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

[CLEANUP]
max_age_days = 7
cleanup_interval = 24
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
    assert cfg["classifier"] == "dfne"
    assert cfg["speciesnet_model"] == "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
    assert cfg["crop_padding"] == 0.15
    assert cfg["motion_threshold"] == 0.02
    assert cfg["motion_check_interval"] == 3


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


def test_classifier_and_speciesnet_model_default_when_keys_absent(tmp_path):
    # classifier/speciesnet_model postdate every existing ac.cfg, so
    # they must default rather than be required -- see the comment in
    # load_config().
    assert "classifier" not in VALID_CFG
    path = write_cfg(tmp_path, VALID_CFG)

    cfg = ac.load_config(path)

    assert cfg["classifier"] == "dfne"
    assert cfg["speciesnet_model"] == "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"


def test_speciesnet_classifier_with_custom_model_path_is_accepted(tmp_path):
    cfg_with_speciesnet = VALID_CFG.replace(
        "static_tolerance = 0.03\n",
        "static_tolerance = 0.03\n"
        "classifier = speciesnet\n"
        "speciesnet_model = /opt/speciesnet_model\n")
    path = write_cfg(tmp_path, cfg_with_speciesnet)

    cfg = ac.load_config(path)

    assert cfg["classifier"] == "speciesnet"
    assert cfg["speciesnet_model"] == "/opt/speciesnet_model"


def test_speciesnet_classifier_without_explicit_model_uses_default(tmp_path):
    # classifier=speciesnet alone (no speciesnet_model override) should
    # work, falling back to Google's own recommended kaggle: identifier
    # rather than requiring every speciesnet user to also set this.
    cfg_with_speciesnet = VALID_CFG.replace(
        "static_tolerance = 0.03\n",
        "static_tolerance = 0.03\nclassifier = speciesnet\n")
    path = write_cfg(tmp_path, cfg_with_speciesnet)

    cfg = ac.load_config(path)

    assert cfg["classifier"] == "speciesnet"
    assert cfg["speciesnet_model"] == "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"


def test_invalid_classifier_value_raises_friendly_systemexit(tmp_path):
    cfg_with_bad_classifier = VALID_CFG.replace(
        "static_tolerance = 0.03\n",
        "static_tolerance = 0.03\nclassifier = bogus\n")
    path = write_cfg(tmp_path, cfg_with_bad_classifier)

    with pytest.raises(SystemExit) as exc_info:
        ac.load_config(path)

    message = str(exc_info.value)
    assert "bogus" in message
    assert "ac.cfg.example" in message


def with_camera_keys(extra):
    """VALID_CFG with extra lines appended to its [CAMERA] section."""
    return VALID_CFG.replace("port = 8554\n", f"port = 8554\n{extra}")


def test_channels_and_rtsp_path_default_to_the_original_hardcoded_layout(
        tmp_path):
    # These keys postdate every existing ac.cfg, and the defaults must be
    # exactly what v0.14 hardcoded -- otherwise upgrading silently points
    # an existing deployment at different (or no) streams.
    assert "channels" not in VALID_CFG
    assert "rtsp_path" not in VALID_CFG

    cfg = ac.load_config(write_cfg(tmp_path, VALID_CFG))

    assert cfg["channels"] == [4, 5, 6]
    assert cfg["rtsp_path"] == "/Streaming/Channels/{channel}02"


def test_custom_channels_and_rtsp_path_are_parsed(tmp_path):
    # Whitespace and a trailing comma are tolerated, and a query-string
    # path (with its own '=' and '&') survives configparser intact.
    path = write_cfg(tmp_path, with_camera_keys(
        "channels = 1, 2 ,3,\n"
        "rtsp_path = /cam/realmonitor?channel={channel}&subtype=1\n"))

    cfg = ac.load_config(path)

    assert cfg["channels"] == [1, 2, 3]
    assert cfg["rtsp_path"] == "/cam/realmonitor?channel={channel}&subtype=1"


def test_single_channel_may_use_a_fixed_path_without_placeholder(tmp_path):
    # A standalone camera has one fixed stream path and no channel
    # concept; the lone channel number then only names its folder.
    path = write_cfg(tmp_path, with_camera_keys(
        "channels = 1\nrtsp_path = /stream2\n"))

    cfg = ac.load_config(path)

    assert cfg["channels"] == [1]
    assert cfg["rtsp_path"] == "/stream2"


def test_zero_padded_format_spec_in_rtsp_path_is_accepted(tmp_path):
    path = write_cfg(tmp_path, with_camera_keys(
        "channels = 1,2,10\nrtsp_path = /live_{channel:02d}_main\n"))

    cfg = ac.load_config(path)

    assert cfg["channels"] == [1, 2, 10]


@pytest.mark.parametrize("extra, key", [
    ("channels = four,five\n", "channels"),
    ("channels =\n", "channels"),
    ("channels = 4,5,4\n", "channels"),
    ("rtsp_path = stream1\n", "rtsp_path"),
    ("rtsp_path = /live/{ch}\n", "rtsp_path"),
    ("rtsp_path = /live/{channel\n", "rtsp_path"),
    # Default channels are 4,5,6; without a placeholder all three would
    # open the same stream.
    ("rtsp_path = /live/main\n", "rtsp_path"),
])
def test_invalid_camera_keys_raise_friendly_systemexit(tmp_path, extra, key):
    path = write_cfg(tmp_path, with_camera_keys(extra))

    with pytest.raises(SystemExit) as exc_info:
        ac.load_config(path)

    message = str(exc_info.value)
    assert key in message
    assert "ac.cfg.example" in message


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
