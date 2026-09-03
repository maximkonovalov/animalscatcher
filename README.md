Animals Catcher is a lightweight, AI-powered surveillance tool.
It processes RTSP camera streams to detect and identify wildlife,
people, and vehicles in real-time.

By utilizing a two-stage AI pipeline, it first detects broad categories
and then performs taxonomic classification to identify specific animal species.

Developed with [Claude](https://claude.com), Anthropic's AI assistant,
using Claude Code and the Claude Sonnet 5 model.

## Core Features

Two-Stage AI Pipeline:

* Stage 1: [MegaDetectorV6](https://github.com/agentmorris/MegaDetector)
        (YOLOv9-C) for high-speed object detection.

* Stage 2: [DFNE](https://code.usgs.gov/vtcfwru/deepfaune-new-england)
        ("Deepfaune-New-England", a USGS retrain of the French
        Deepfaune model on northeastern US species) for species
        identification (Coyotes, Bobcats, Black Bear, etc.).

* Smart Alerts: Sends labeled snapshots to a Telegram bot with configurable
cooldowns to prevent notification flooding.

* Auto-Maintenance: Integrated cleanup engine to purge old snapshots
and manage disk space automatically.

* Persistent Monitoring: Multi-threaded architecture with auto-reconnect
logic for unstable RTSP streams.

## System Requirements

Environment: Python 3.10 (verified). PytorchWildlife's yolov5
dependency is incompatible with Python 3.12 -- it imports pkg_resources
in a way that breaks on 3.12 either direction: pkgutil.ImpImporter
(which old setuptools' pkg_resources needs) was removed in 3.12, but
upgrading setuptools to fix that instead removes pkg_resources itself
in newer releases. requests>=2.33 (see requirements.txt) also requires
Python>=3.10. Untested on 3.11.

Dependencies: opencv-python-headless, PytorchWildlife, requests,
configparser.

Hardware: i3 CPU or better; requires internet access for initial model
downloads.

## Configuration (ac.cfg)

The program relies on an external configuration file. Ensure the
following sections are defined:

| Section   | Keys | Description |
| :---      | :--- |        :--- |
| CAMERA    | user, pass, ip, port | RTSP credentials and network address. |
| TELEGRAM  | token, chat_id | Bot API token and target chat ID for alerts. |
| PATHS     | base_output_folder | Storage location for snapshots. |
| DETECTION | threshold_0-2, cooldown, frame_interval, summary_interval, species_threshold, static_tolerance | Confidence thresholds and alert frequency. |
| CLEANUP | max_age_days, cleanup_interval | Retention policy for snapshots. |

`ac.cfg` holds RTSP and Telegram credentials in plaintext and must never be
committed (it's covered by `.gitignore`). Since the daemon typically runs as
root via launchd, lock the file down to the owning user:

    chmod 600 ac.cfg

## Installation & Usage

Deploy Code: Place ac.py and ac.cfg in a working directory..

Install Dependencies:

`PytorchWildlife` pulls in `torch` as a transitive dependency. On Linux
and Windows, a plain `pip install` grabs the CUDA-enabled build by
default, which is several gigabytes even though this project only runs
on CPU. Install the CPU-only build first to avoid that download
(macOS wheels are already CPU/MPS-only, so this step is unnecessary there):

    pip install torch --index-url https://download.pytorch.org/whl/cpu

Then install the pinned project dependencies:

    pip install -r requirements.txt

Run the Daemon:

    python3 ac.py

Optional macOS Deployment: `com.user.ac.plist` and `deploy.sh`
automate running this as a launchd LaunchDaemon on the machine this
project was developed against. Both hardcode paths and a username
specific to that deployment (`/Users/maxim/nvr` as the project
directory, `/opt/local/bin/python3.10` as the MacPorts interpreter,
and `UserName maxim` as the account the daemon drops root privileges
to) -- edit these to match your own machine and account before using
them; they aren't meant to work as-is elsewhere.

Monitor: Check stdout (or wherever your process supervisor captures it
-- e.g. StandardOutPath in com.user.ac.plist) or your Telegram channel
for the "The Animals Catcher is online" startup message. All of this
daemon's own logging goes to stdout; nothing is written to a separate
app log file.

## Project Structure

ai_engine: The brain of the system; handles detection and classification.

camera_thread: Manages RTSP streams for channels 4, 5, and 6.

summary_engine: Sends periodic health reports and detection stats to Telegram.

cleanup_engine: Keeps the system lean by purging old data.

Note: On the first execution, the system will download approximately
300MB of AI model weights. Ensure a stable connection is available.

## Development

Install the pinned dev tooling, lint, and run the tests before committing
(use a Python 3.10 interpreter -- see System Requirements above):

    pip install -r requirements-dev.txt
    ruff check ac.py
    pip install -r requirements.txt   # tests import ac.py directly
    pytest tests/

`ruff.toml` pins the project's actual lint rule selection (and
documents why a couple of ruff's newer default rules -- broad
`except Exception`, naive `datetime.now()` -- are deliberately not
enabled here), so `ruff check ac.py` gives the same result regardless
of which ruff version happens to be installed.

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the same
lint/syntax checks and the test suite (on Python 3.10) on every push
and pull request.

### EOF
