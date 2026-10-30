I failed to find a ready to use solution that parses my home CCTV
streams and detects wildlife and also I wanted to play with various
ML/AI tools so I decided to re-invent my own wheel. This is a free
and open source software under BSD license. I run it on my ancient
Mac Mini 2018 but the code itself should run on any UNIX-like systems.

Animals Catcher is a lightweight, AI-powered surveillance tool.
It processes RTSP camera streams to detect and identify wildlife,
people, and vehicles in real-time.

By utilizing a two-stage AI pipeline, it first detects broad categories
and then performs taxonomic classification to identify specific animal species.

Developed with [Claude](https://claude.com), Anthropic's AI assistant,
using Claude Code and the Claude Sonnet 5 model.

![Sample detection: two mule deer identified by the two-stage AI pipeline, with species and confidence labels](assets/sample-detection.jpg)

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
| CAMERA    | user, pass, ip, port, channels, rtsp_path | RTSP credentials, network address, and which streams to open (see Camera Setup below). |
| TELEGRAM  | token, chat_id | Bot API token and target chat ID for alerts. |
| PATHS     | base_output_folder | Storage location for snapshots. |
| DETECTION | threshold_0-2, cooldown, frame_interval, motion_threshold, motion_check_interval, summary_interval, species_threshold, static_tolerance, classifier, speciesnet_model, crop_padding | Confidence thresholds, alert frequency, and species classifier choice. |
| CLEANUP | max_age_days, cleanup_interval | Retention policy for snapshots. |

`ac.cfg` holds RTSP and Telegram credentials in plaintext and must never be
committed (it's covered by `.gitignore`). Since the daemon runs via launchd
under whichever account `UserName` in `assets/com.user.ac.plist` specifies (or root,
if that key is absent), lock the file down to the owning user:

    chmod 600 ac.cfg

## Camera Setup

Each stream is opened as `rtsp://<user>:<pass>@<ip>:<port><rtsp_path>`:
`user`, `pass`, `ip` and `port` come from `[CAMERA]` (the credentials are
URL-escaped for you), and `rtsp_path` is the rest of the URL, with
`{channel}` replaced by each number in `channels`. The daemon runs one
capture thread per listed channel and saves its snapshots under a folder
named after it (`cam04`, `cam05`, ...).

Both keys are optional. Their defaults are the layout this project was
originally hardcoded for -- a Hikvision-style DVR with cameras on
channels 4, 5 and 6, each read from its sub-stream (`N02`, the
lower-resolution one, which is cheaper to decode) -- so an existing
`ac.cfg` keeps working unchanged. For other setups, for example:

    # Hikvision-style recorder, channels 1-4, full-resolution main streams
    channels = 1,2,3,4
    rtsp_path = /Streaming/Channels/{channel}01

    # Dahua/Amcrest-style, query-string URL, sub-stream
    channels = 1,2
    rtsp_path = /cam/realmonitor?channel={channel}&subtype=1

    # One standalone camera with a fixed stream path: the single channel
    # number only names the snapshot folder (cam01)
    channels = 1
    rtsp_path = /stream2

These illustrate the pattern; they aren't a compatibility list -- look
up the RTSP path in your camera's or recorder's manual. `{channel}`
accepts Python format specs, so `{channel:02d}` gives zero-padded numbers
(`01`, `02`, ...) for recorders whose paths expect them. `rtsp_path` must
start with `/`, and must contain `{channel}` whenever more than one
channel is listed (otherwise every thread would open the same stream);
`ac.py` refuses to start, naming the key, if either rule is broken. All
channels share one `ip`, `port` and login, so cameras on different hosts
or with different credentials can't be mixed in a single instance.

To check a URL before starting the daemon, try it with `ffprobe` (the
daemon also forces RTSP over TCP):

    ffprobe -rtsp_transport tcp "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/102"

## Installation & Usage

Deploy Code: Place ac.py and ac.cfg in a working directory. By default
`ac.py` looks for `ac.cfg` next to itself; set `AC_CONFIG_PATH` to point
it at a config file somewhere else instead (e.g. to run more than one
instance from the same checkout).

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

Optional macOS Deployment: `assets/com.user.ac.plist` and `deploy.sh`
automate running this as a launchd LaunchDaemon on the machine this
project was developed against. Both hardcode paths and a username
specific to that deployment (`/Users/maxim/nvr` as the project
directory, `/opt/local/bin/python3.10` as the MacPorts interpreter,
and `UserName maxim` as the account the daemon drops root privileges
to) -- edit these to match your own machine and account before using
them; they aren't meant to work as-is elsewhere.

Monitor: Check stdout (or wherever your process supervisor captures it
-- e.g. StandardOutPath in assets/com.user.ac.plist) or your Telegram channel
for the "The Animals Catcher is online" startup message. All of this
daemon's own logging goes to stdout; nothing is written to a separate
app log file.

Log Rotation: since the daemon logs to stdout/stderr rather than a
file it manages itself, this project doesn't rotate its own logs --
`assets/com.user.ac.newsyslog.conf` (installed to `/etc/newsyslog.d/` by
deploy.sh) hands that job to macOS's own newsyslog. Renaming a log
file out from under a running process doesn't do anything useful on
its own though: launchd only opens StandardOutPath/StandardErrorPath
once, at process start, so this daemon's inherited fds would just keep
appending to the newly-archived file forever. newsyslog's pid_file and
signal_number config fields close that gap: after rotating, it sends
SIGUSR1 to the PID in `ac.pid` (written at startup for this purpose),
which `ac.py`'s own signal handler uses to reopen fresh file
descriptors at the same two paths (via `AC_STDOUT_LOG`/`AC_STDERR_LOG`,
set in assets/com.user.ac.plist) -- no daemon restart needed.

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

camera_thread: Manages one RTSP stream per channel in `[CAMERA] channels`
(see Camera Setup above).

summary_engine: Sends periodic health reports and detection stats to Telegram.

cleanup_engine: Keeps the system lean by purging old data.

Three engine threads (ai_engine, summary_engine, cleanup_engine) plus one
camera_thread per configured channel run for the life of the process --
6 in total with the default channels (4, 5, 6). The main thread just
starts them and idles.

The diagram below traces a single frame from capture through to a
Telegram alert, and shows summary_engine and cleanup_engine running
independently alongside it, reading/writing the same stats dict and
output folder:

```mermaid
flowchart TD
    C["camera_thread<br/>(one per configured channel)"] -->|sampled frame| Q

    Q[("detection_queue<br/>priority: motion frames before frame_interval frames")] --> AI(["ai_engine"])

    AI --> MD["Stage 1: MegaDetector<br/>per-class threshold filter (Animal/Person/Vehicle)"]
    MD -->|"class Animal, conf > species_threshold"| SC["Stage 2: species classifier<br/>(DFNE or SpeciesNet)"]
    MD -->|"class Person / Vehicle"| Gate
    SC --> Gate["Alert decision gate<br/>(two-frame latch, see below)"]

    Gate -->|pass| Snap["write annotated snapshot"]
    Gate -->|fail| NoOp["no alert"]
    Snap --> TG(["Telegram photo alert<br/>(async via photo_executor)"])
    Snap --> Out[("base_output_folder")]

    Stats[("stats dict")]
    C -.-> Stats
    AI -.-> Stats

    Sum["summary_engine<br/>(every summary_interval hours)"] -->|health + detection report| TG2(["Telegram report"])
    Stats -.->|read, then reset| Sum

    Clean["cleanup_engine<br/>(every cleanup_interval hours)"] -->|delete files older than max_age_days| Out
```

The alert decision gate isn't a single-frame decision: a class has to
show up on two consecutive processed frames for the same (cam_id, class)
pair, with cooldown elapsed and the box not flagged as static/stuck,
before Telegram actually gets a photo. Cooldown and the static filter
are tracked independently per camera/class pair, not globally:

```mermaid
flowchart TD
    A["Detection for (cam_id, class)<br/>on this processed frame"] --> B{"Was this (cam_id, class)<br/>also seen on the PREVIOUS<br/>processed frame?"}
    B -->|no| L["Latch only: record seen,<br/>no alert yet"]
    B -->|yes| C{"time since last_det[cam_id, class]<br/>> cooldown?"}
    C -->|no| D["Suppressed: still in cooldown"]
    C -->|yes| E{"box position vs. last_box<br/>within static_tolerance?"}
    E -->|yes, static/stuck box| F["Filtered: logged, no alert"]
    E -->|no, box moved| G["Real detection: update stats,<br/>last_box and last_det;<br/>snapshot + Telegram alert"]
```

Note: On the first execution, the system will download approximately
300MB of AI model weights. Ensure a stable connection is available.

## Development

Install the pinned dev tooling, lint, and run the tests before committing
(use a Python 3.10 interpreter -- see System Requirements above):

    pip install -r requirements-dev.txt
    ruff check ac.py

Tests import `ac.py` directly, so they need the same runtime
dependencies as a real deployment -- follow the CPU-only torch install,
`pip install -r requirements.txt`, and opencv-python-headless
force-reinstall steps from Installation & Usage above, then:

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
