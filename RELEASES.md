# Release History

## v0.5 — 2026-09-02

A major reliability, security, and code-quality pass across the whole
project, covering bug fixes, operational hardening, dependency
packaging, and developer tooling.

### Fixed
- `OPENCV_FFMPEG_CAPTURE_OPTIONS` was missing the `|` separator between
  FFmpeg `key;value` pairs, so the RTSP read timeout (`stimeout`) never
  actually applied and a stalled stream could hang indefinitely.
- RTSP username/password were not URL-escaped, breaking the connection
  whenever credentials contained `@`, `:`, `/`, or `#`.
- Species-classification confidence used an `or`-chain fallback
  (`res.get('confidence') or res.get('y_conf') or 0.0`), which
  incorrectly discarded a legitimate `0.0` confidence value.

### Added
- `species_threshold` and `static_tolerance` moved into `ac.cfg` under
  `[DETECTION]` (previously hardcoded), consistent with every other
  tunable.
- Thread-safe, self-rotating logging via
  `logging.handlers.RotatingFileHandler`, replacing hand-rolled
  open/append/truncate log writes.
- Crash isolation: `ai_engine`, `camera_thread`, `cleanup_engine`, and
  `summary_engine` now log and continue on an unexpected error instead
  of dying silently and permanently as a daemon thread.
- Fail-fast startup check if `ac.cfg` is missing or unreadable.
- Bounded Telegram photo uploads via a 4-worker `ThreadPoolExecutor`,
  replacing an unbounded thread spawned per detection.
- `SIGTERM`/`SIGINT` handling for a logged, graceful shutdown.
- Named background threads (`ai_engine`, `cam04`/`cam05`/`cam06`, etc.)
  for easier debugging.
- `ThrottleInterval` added to the launchd plist, so a persistent
  AI-model load failure can't turn into a restart storm.
- Date (`DD/MM/YYYY`) added to the periodic summary and startup
  Telegram messages; the startup message now also reports the number
  of configured streams.
- `.gitignore`, pinned versions in `requirements.txt`, a new
  `requirements-dev.txt` (pinning `ruff`), and a GitHub Actions lint
  workflow (`.github/workflows/lint.yml`) running `ruff` and
  `py_compile` on every push.
- Documentation: `ac.cfg` permission guidance, CPU-only `torch`
  install instructions, and a Development section in the README.

### Changed
- Replaced every bare `except: pass` with specific exception handling
  and logging.
- Style cleanup: removed single-line compound statements and
  inconsistent operator/comma spacing (verified with `ruff`).
- Shebang changed from the MacPorts-specific `/opt/local/bin/python3`
  to `#!/usr/bin/env python3`.

### Breaking
- Existing `ac.cfg` files must add `species_threshold` and
  `static_tolerance` under `[DETECTION]` (see `ac.cfg.example`) or the
  daemon will fail to start.

## v0.4 — 2026-03-29

- Lowered the default species-classification threshold (0.6 → 0.45).
- Added defensive key lookups for the classifier's output
  (`label`/`prediction`/`y_pred`, `confidence`/`y_conf`) after the
  upstream schema changed; superseded by a more correct fix in v0.5.
- Added debug logging around classifier errors.

## v0.3 — 2026-03-26

- Fixed the species-labeling logic.
- Added `requirements.txt` for dependency installation.

## v0.2 — 2026-03-25

- Corrected the startup notification message.
- Relocated log file storage.
- Removed macOS-specific assumptions from setup instructions.
- README formatting and clarity improvements.
- Added the macOS `launchctl` command cheat sheet
  (`launchctl.txt`).

## v0.1 (unreleased / initial commit) — 2026-03-19

- Initial implementation: RTSP capture across multiple channels,
  two-stage MegaDetector + DeepFaune detection/classification
  pipeline, Telegram alerting, and macOS launchd deployment scaffolding
  (`deploy.sh`, `com.user.ac.plist`).
