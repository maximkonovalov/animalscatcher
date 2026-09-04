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

* Stage 2: species identification, via a configurable classifier --
        [DFNE](https://code.usgs.gov/vtcfwru/deepfaune-new-england)
        (default; "Deepfaune-New-England", a USGS retrain of the French
        Deepfaune model on northeastern US species) or
        [SpeciesNet](https://github.com/google/cameratrapai) (Google's
        2498-taxa classifier, for regions DFNE doesn't cover well). See
        Species Classifier below.

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
| DETECTION | threshold_0-2, cooldown, frame_interval, summary_interval, species_threshold, static_tolerance, classifier, speciesnet_model, crop_padding | Confidence thresholds, alert frequency, and species classifier choice. |
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

PytorchWildlife's own dependencies (`ultralytics`, and `yolov5` via
`sahi`) unconditionally require plain `opencv-python`, which gets
installed alongside the `opencv-python-headless` pinned above and
silently wins the `cv2` import (confirmed via `cv2.__version__` --
`pip check` alone won't catch this, since both packages install
without error). Force `opencv-python-headless` back to being the one
actually used:

    pip uninstall -y opencv-python
    pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84

(`pip check` will then report sahi/ultralytics/yolov5 wanting
`opencv-python` -- expected and harmless: headless provides everything
those libraries actually call, just not GUI functions like
`cv2.imshow`, which this project never uses either.)

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

Log Rotation: since the daemon logs to stdout/stderr rather than a
file it manages itself, this project doesn't rotate its own logs --
`com.user.ac.newsyslog.conf` (installed to `/etc/newsyslog.d/` by
deploy.sh) hands that job to macOS's own newsyslog. Renaming a log
file out from under a running process doesn't do anything useful on
its own though: launchd only opens StandardOutPath/StandardErrorPath
once, at process start, so this daemon's inherited fds would just keep
appending to the newly-archived file forever. newsyslog's pid_file and
signal_number config fields close that gap: after rotating, it sends
SIGUSR1 to the PID in `ac.pid` (written at startup for this purpose),
which `ac.py`'s own signal handler uses to reopen fresh file
descriptors at the same two paths (via `AC_STDOUT_LOG`/`AC_STDERR_LOG`,
set in com.user.ac.plist) -- no daemon restart needed.

## Species Classifier

Stage 2 (species identification) is configurable via `[DETECTION]
classifier` in `ac.cfg`:

* `dfne` (default): [DFNE](https://code.usgs.gov/vtcfwru/deepfaune-new-england),
  needs only `requirements.txt`. Its species set is northeastern-US
  (bobcat, coyote, black bear, gray/red fox, raccoon, skunk,
  white-tailed deer, wild turkey, etc.) -- no changes needed for that
  region.
* `speciesnet`: [Google's SpeciesNet](https://github.com/google/cameratrapai),
  2498 taxa, geographically broad -- better suited to regions DFNE
  doesn't cover well (e.g. no puma/mountain lion in DFNE at all, and
  mule deer vs. DFNE's white-tailed deer only). Also needs
  `pip install -r requirements-speciesnet.txt` (a separate, much
  heavier dependency set -- pandas, matplotlib, kagglehub, ... -- kept
  out of the base install since most deployments won't use it).

For `speciesnet`, `speciesnet_model` (also in `[DETECTION]`) selects
which model to load -- defaults to Google's own recommended
`kaggle:google/speciesnet/pyTorch/v4.0.3a/1`, downloaded automatically
via `kagglehub` on first use (no Kaggle account needed for this public
model; confirmed by actually downloading it -- expect ~215MB and
20-40s to load on CPU). Set it to a local directory instead (one
already containing a previously-downloaded copy) for an
offline-friendly deployment.

`requirements-speciesnet.txt` reintroduces the same `opencv-python`
leak documented above (SpeciesNet also depends on `yolov5`) -- re-run
the `pip uninstall opencv-python` / `--force-reinstall --no-deps
opencv-python-headless` fix after installing it, same as after
`requirements.txt` alone.

Only the bare classifier is used here (single-crop inference on
MegaDetector's already-cropped detections), not SpeciesNet's own
CLI/ensemble pipeline -- so its geographic geofencing (e.g. restricting
predictions to species actually found in California) isn't applied;
every prediction is the classifier's raw top-1 guess across all 2498
taxa.

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
