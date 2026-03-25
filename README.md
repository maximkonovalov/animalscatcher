Animal Catcher is a lightweight, AI-powered surveillance tool designed
for macOS. It processes RTSP camera streams to detect and identify wildlife,
people, and vehicles in real-time.

By utilizing a two-stage AI pipeline, it first detects broad categories
and then performs taxonomic classification to identify specific animal species.

## Core Features

Two-Stage AI Pipeline:

* Stage 1: MegaDetectorV6 (YOLOv9-C) for high-speed object detection.

* Stage 2: DeepfauneClassifier for species identification (Coyotes,
        Bobcats, etc.).

* Smart Alerts: Sends labeled snapshots to a Telegram bot with configurable
cooldowns to prevent notification flooding.

* Auto-Maintenance: Integrated cleanup engine to manage disk space and
log file sizes automatically.

* Persistent Monitoring: Multi-threaded architecture with auto-reconnect
logic for unstable RTSP streams.

## System Requirements

OS: macOS (optimized for /opt/local/bin/python3).

Environment: Python 3.12+.

Dependencies: opencv-python, PytorchWildlife, requests, configparser.

Hardware: i3 CPU or better; requires internet access for initial model
downloads.

## Configuration (ac.cfg)

The program relies on an external configuration file. Ensure the
following sections are defined:

| Section   | Keys | Description |
| :---      | :--- |        :--- |
| CAMERA    | user, pass, ip, port | RTSP credentials and network address. |
| TELEGRAM  | token, chat_id | Bot API token and target chat ID for alerts. |
| PATHS     | base_output_folder, log_file | Storage locations for snapshots and system logs. |
| DETECTION | threshold_0-2, cooldown | Confidence thresholds and alert frequency. |
| CLEANUP | max_age_days, max_log_size_mb | Retention policies for data management. |

## Installation & Usage

Deploy Code: Place ac.py and ac.cfg in a working directory..

Install Dependencies:

```bash
pip install -r requirements.txt```

Run the Daemon:

```bash
python3 ac.py```

Monitor: Check the log file defined in your config or your Telegram
channel for the "An animal catcher is online" startup message.

## Project Structure

ai_engine: The brain of the system; handles detection and classification.

camera_thread: Manages RTSP streams for channels 4, 5, and 6.

summary_engine: Sends periodic health reports and detection stats to Telegram.

cleanup_engine: Keeps the system lean by purging old data.

Note: On the first execution, the system will download approximately
300MB of AI model weights. Ensure a stable connection is available.

### EOF
