"""Shared pytest fixtures.

`ac.py` loads and validates its config file at import time. Since a real
ac.cfg is intentionally gitignored (it holds RTSP/Telegram credentials)
and normally lives next to ac.py, we point the module at a throwaway,
valid config via the AC_CONFIG_PATH env var *before* any test module
imports ac -- this lets `import ac` succeed in a clean checkout without
touching (or requiring) a real config file.
"""
import os
import sys
import tempfile

_FIXTURE_DIR = tempfile.mkdtemp(prefix="animalscatcher-test-")
_FIXTURE_CFG = os.path.join(_FIXTURE_DIR, "ac.cfg")

with open(_FIXTURE_CFG, "w") as f:
    f.write(f"""\
[CAMERA]
user = testuser
pass = testpass
ip = 127.0.0.1
port = 8554

[TELEGRAM]
token = test-token
chat_id = -100

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
base_output_folder = {_FIXTURE_DIR}/snaps
log_file = {_FIXTURE_DIR}/ac_log.txt

[CLEANUP]
max_age_days = 7
cleanup_interval = 24
max_log_size_mb = 5
""")

os.environ.setdefault("AC_CONFIG_PATH", _FIXTURE_CFG)

# Make the repo root (one level up from tests/) importable so `import ac`
# finds ac.py regardless of where pytest is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
