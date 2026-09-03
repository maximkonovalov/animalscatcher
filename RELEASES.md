Release History
===============

v0.7 (unreleased, no tag) - 2026-09-02
----------------------------------------

Security fix, dependency automation, and a first test suite. Not
tagged as a release per the maintainer's request, but VERSION in
ac.py reflects it.

Fixed:
  - Telegram bot token could leak into ac_log.txt: requests
    exceptions (ConnectionError, Timeout) often stringify the full
    request URL, which embeds the token. Added _redact() and applied
    it to every logged Telegram exception.

Added:
  - .github/dependabot.yml: weekly pip and GitHub Actions update PRs.
  - tests/ with pytest: config validation (valid config, missing
    file/key/section, invalid numeric value), the token-redaction
    helper, and mocked-detector coverage of the debounce/static-filter/
    cooldown alert logic plus the stats[obj_name] KeyError regression.
  - Refactored config loading into load_config(path), a pure function
    returning a typed dict, with an AC_CONFIG_PATH env override so
    tests can point ac.py at a fixture config. Also fixed a gap where
    an invalid numeric value (e.g. cooldown = not-a-number) escaped as
    an uncaught ValueError instead of the friendly SystemExit that
    missing keys already got.
  - .github/workflows/ci.yml (renamed from lint.yml): added a `test`
    job running the suite alongside the existing lint job.
  - requirements-dev.txt: pytest==8.4.2.

Changed:
  - Switched opencv-python to opencv-python-headless: ac.py never uses
    a GUI feature, and the non-headless build needs libGL/GTK/X11
    system libraries a headless server or CI image won't have.
    PytorchWildlife still pulls in plain opencv-python transitively,
    so deploy.sh and ci.yml both force-reinstall the headless build
    after the main install to make sure it wins the shared `cv2` path.
  - CI (and the README's stated System Requirements) target Python
    3.10, not 3.12: PytorchWildlife==1.1.1's yolov5 dependency needs
    pkg_resources in a way that has no compatible setuptools version
    on 3.12 (old setuptools' pkgutil.ImpImporter was removed; newer
    setuptools instead removes pkg_resources itself).

v0.6 - 2026-09-02
------------------

Fixes from an independent multi-angle code review of the v0.5 changes.

Fixed:
  - Config validation now covers a missing key, not just a missing
    file: all config reads are wrapped in try/except configparser.Error,
    raising a friendly message instead of an unhandled traceback with
    no log record.
  - Restored the "never raises" contract on the Telegram helper
    functions (except Exception, not the narrower RequestException) --
    the unprotected startup announcement in ai_engine could otherwise
    be killed by any non-network exception, permanently halting all
    detection with no restart.
  - Snapshot filenames now include the object class, fixing a
    same-second collision where two different classes detected in one
    frame could overwrite each other's saved photo before the async
    upload read it.
  - Narrowed camera_thread's stats_lock scope to just the dict writes,
    no longer held across the blocking reconnect (release/sleep/
    VideoCapture), which could stall every other camera thread and
    summary_engine.
  - Fixed a latent KeyError in stats[obj_name] for any detector class
    outside {Animal, Person, Vehicle}.
  - Applied the same not-None fallback already used for species
    confidence to the species label, removing the matching latent bug.

Changed:
  - photo_executor is now explicitly shut down on SIGTERM/SIGINT
    instead of relying on the implicit ThreadPoolExecutor atexit join.
  - Added a 1s backoff to ai_engine's error path, matching the
    throttle camera_thread already has, to avoid log spam on a
    persistent error.
  - deploy.sh now installs from the pinned requirements.txt instead of
    a separate unpinned package list.

v0.5 - 2026-09-02
------------------

A major reliability, security, and code-quality pass across the whole
project, covering bug fixes, operational hardening, dependency
packaging, and developer tooling.

Fixed:
  - OPENCV_FFMPEG_CAPTURE_OPTIONS was missing the "|" separator between
    FFmpeg key;value pairs, so the RTSP read timeout (stimeout) never
    actually applied and a stalled stream could hang indefinitely.
  - RTSP username/password were not URL-escaped, breaking the
    connection whenever credentials contained @, :, /, or #.
  - Species-classification confidence used an or-chain fallback
    (res.get('confidence') or res.get('y_conf') or 0.0), which
    incorrectly discarded a legitimate 0.0 confidence value.

Added:
  - species_threshold and static_tolerance moved into ac.cfg under
    [DETECTION] (previously hardcoded), consistent with every other
    tunable.
  - Thread-safe, self-rotating logging via
    logging.handlers.RotatingFileHandler, replacing hand-rolled
    open/append/truncate log writes.
  - Crash isolation: ai_engine, camera_thread, cleanup_engine, and
    summary_engine now log and continue on an unexpected error instead
    of dying silently and permanently as a daemon thread.
  - Fail-fast startup check if ac.cfg is missing or unreadable.
  - Bounded Telegram photo uploads via a 4-worker ThreadPoolExecutor,
    replacing an unbounded thread spawned per detection.
  - SIGTERM/SIGINT handling for a logged, graceful shutdown.
  - Named background threads (ai_engine, cam04/cam05/cam06, etc.) for
    easier debugging.
  - ThrottleInterval added to the launchd plist, so a persistent
    AI-model load failure can't turn into a restart storm.
  - Date (DD/MM/YYYY) added to the periodic summary and startup
    Telegram messages; the startup message now also reports the number
    of configured streams.
  - .gitignore, pinned versions in requirements.txt, a new
    requirements-dev.txt (pinning ruff), and a GitHub Actions lint
    workflow (.github/workflows/lint.yml) running ruff and py_compile
    on every push.
  - Documentation: ac.cfg permission guidance, CPU-only torch install
    instructions, and a Development section in the README.

Changed:
  - Replaced every bare "except: pass" with specific exception
    handling and logging.
  - Style cleanup: removed single-line compound statements and
    inconsistent operator/comma spacing (verified with ruff).
  - Shebang changed from the MacPorts-specific /opt/local/bin/python3
    to #!/usr/bin/env python3.

Breaking:
  - Existing ac.cfg files must add species_threshold and
    static_tolerance under [DETECTION] (see ac.cfg.example) or the
    daemon will fail to start.

v0.4 - 2026-03-29
------------------

  - Lowered the default species-classification threshold (0.6 -> 0.45).
  - Added defensive key lookups for the classifier's output
    (label/prediction/y_pred, confidence/y_conf) after the upstream
    schema changed; superseded by a more correct fix in v0.5.
  - Added debug logging around classifier errors.

v0.3 - 2026-03-26
------------------

  - Fixed the species-labeling logic.
  - Added requirements.txt for dependency installation.

v0.2 - 2026-03-25
------------------

  - Corrected the startup notification message.
  - Relocated log file storage.
  - Removed macOS-specific assumptions from setup instructions.
  - README formatting and clarity improvements.
  - Added the macOS launchctl command cheat sheet (launchctl.txt).

v0.1 (unreleased / initial commit) - 2026-03-19
-------------------------------------------------

  - Initial implementation: RTSP capture across multiple channels,
    two-stage MegaDetector + DeepFaune detection/classification
    pipeline, Telegram alerting, and macOS launchd deployment
    scaffolding (deploy.sh, com.user.ac.plist).
