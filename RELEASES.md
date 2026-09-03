Release History
===============

v0.10 - 2026-09-03
------------------

Documentation polish, no functional code changes:

  - Removed debug/rtsp_probe.py and its throwaway plists now that the
    UserName maxim privilege drop (v0.9) has been confirmed working
    end-to-end, including through a full machine reboot.
  - AGENTS.md still described the classifier as plain "DeepFaune" --
    the same species-list mixup (no coyote/bobcat) v0.8 specifically
    fixed everywhere else. Corrected to DFNE, and documented two
    behaviors added in v0.9 that AGENTS.md hadn't caught up on yet:
    per-detection confidence logging and the stale partial-download
    cleanup at ai_engine() startup.
  - README now credits Claude (Anthropic), Claude Code, and the Claude
    Sonnet 5 model used to develop this project, and links
    MegaDetectorV6 and DFNE to their actual source projects
    (https://github.com/agentmorris/MegaDetector and
    https://code.usgs.gov/vtcfwru/deepfaune-new-england -- the latter
    pulled directly from PytorchWildlife 1.3.0's own DFNE.py docstring
    rather than guessed).

v0.9 - 2026-09-03
------------------

Version bump only, no other code changes.

Also retried the UserName maxim privilege drop (com.user.ac.plist)
that v0.8 left running as root as an interim, unresolved state:

  - Built a standalone diagnostic (debug/rtsp_probe.py) that isolates
    RTSP/network behavior from AI model loading entirely: reports
    process identity (uid/euid/user), does a raw TCP connect to the
    camera, then a cv2.VideoCapture probe using the exact URL, quoting,
    and FFmpeg options camera_thread() actually uses. Run once as root
    and once as UserName maxim (same signed python3.10) via two
    throwaway LaunchDaemons, so their output could be diffed directly.
  - First run 404'd identically under both identities -- turned out to
    be a bug in the probe itself (a bare rtsp://user:pass@ip:port with
    no path), not a privilege difference; fixed to match ac.py's real
    /Streaming/Channels/{cam_num}02 URL.
  - With the corrected URL, both identities came back identical: TCP
    connect and RTSP frame read both succeeded under maxim, unlike the
    earlier "No route to host" symptom. Restored UserName: maxim in
    com.user.ac.plist on the strength of that result.
  - Along the way: `launchctl bootstrap system` for a new LaunchDaemon
    requires the plist to be owned by root:wheel in a non-writable
    location (e.g. /Library/LaunchDaemons) -- the same requirement
    deploy.sh already handles for com.user.ac.plist. Also, once a
    RunAtLoad-only (no KeepAlive) job has run and exited, a later
    bootstrap of that same plist can fail with a generic "Input/output
    error" even though `kickstart` reports the service isn't loaded;
    an explicit `launchctl bootout` first resolved it.
  - Still not a confirmed fix: the same signing fix looked clean once
    before (v0.8) and then broke again for reasons never isolated, so
    this is being watched closely in production rather than treated as
    resolved. Revert to no UserName (runs as root) in com.user.ac.plist
    if it recurs.

v0.8 - 2026-09-03
------------------

Fixes the AI models entirely failing to load, discovered while
debugging a production deployment on a fresh machine. Traced through
three unrelated layers before finding the real cause.

Fixed:
  - MegaDetectorV6 crashed with "local variable 'url' referenced
    before assignment" on every startup. Root cause: ac.py's
    version="MDV6-yolov9-c" argument was correct for the API of newer
    PytorchWildlife releases, but the project was pinned to 1.1.1
    (for classifier schema stability, back in v0.5), whose
    MegaDetectorV6 only accepts version='yolov9c' or 'rtdetrl' --
    anything else silently falls through without setting a required
    local variable instead of raising. The pin itself was the bug.
  - Species classification (Stage 2) had likely never worked at all
    under the 1.1.1 pin: pw_classification.DeepfauneClassifier doesn't
    exist in that release (confirmed by reading the installed
    package's source directly). Since both model loads happen in the
    same try block, this was masked by the MegaDetectorV6 crash above
    -- fixing that alone would have just swapped one startup crash for
    another.
  - logger.critical() on a model-load failure only logged str(e), not
    a traceback, which is what made the "referenced before assignment"
    error hard to place -- added exc_info=True so future failures show
    exactly which file/line raised.

Changed:
  - Bumped PytorchWildlife 1.1.1 -> 1.3.0. This is also what actually
    fixes the two bugs above: 1.3.0's MegaDetectorV6 accepts the
    version strings ac.py already passes, and it adds DFNE
    ("Deepfaune-New-England", a USGS retrain of Deepfaune) as a real,
    available classifier.
  - Switched the classifier from DeepfauneClassifier to DFNE. This is
    also a correctness fix, not just an availability one: the French
    Deepfaune model's species list has no coyote or bobcat at all --
    it's trained on European fauna (wolf, lynx, chamois, wild boar,
    etc.). DFNE's species list (Bobcat, Coyote, Black Bear, Gray Fox,
    Red Fox, Raccoon, Skunk, White-tailed Deer, Wild Turkey, ...)
    actually matches the species this project's README has always
    claimed to identify. No changes needed to ac.py's existing
    label/prediction/y_pred and confidence/y_conf fallback parsing --
    DFNE's real output already uses the keys that logic expects.
  - requirements.txt gained several new pins that PytorchWildlife 1.3.0
    needs but doesn't declare or that changed transitively: numpy==1.26.4
    (a newer opencv-python/opencv-python-headless pulled in
    transitively requires numpy>=2, which silently breaks
    torch.from_numpy() -- RuntimeError: Numpy is not available -- since
    torch==2.2.2 is compiled against the numpy 1.x ABI), setuptools==
    59.5.0 (same pkg_resources/pkgutil.ImpImporter story as before),
    and soundfile/librosa/numba/llvmlite (PytorchWildlife 1.3.0's
    top-level __init__.py unconditionally imports a new bioacoustics
    submodule that needs these, even though this project never touches
    audio -- an upstream packaging gap, not something optional).
  - With the full pin set above in one requirements.txt, a plain
    `pip install -r requirements.txt` now resolves correctly in one
    shot -- removed the separate `pip install --force-reinstall
    --no-deps opencv-python-headless==...` step from ci.yml and
    deploy.sh that v0.7 needed as a workaround.

Verified end-to-end, three times, from a completely clean venv (CPU
torch -> requirements.txt -> pytest): both MegaDetectorV6 and DFNE
actually load and run real inference (not mocked) with no errors, and
the DFNE output was confirmed to already match ac.py's existing result
parsing.

Also investigated while deploying the above to the real target machine
(same v0.8, no separate version bump -- these are deploy-config/docs
changes, not code). Partially resolved; see UNRESOLVED note below:

  - camera_thread couldn't connect to the RTSP camera at all
    ("No route to host"), which looked like a Python 3.10-vs-3.12
    regression for a long time since the daemon was last confirmed
    working under 3.12. It wasn't: isolated testing (bare socket, then
    cv2.VideoCapture, run as throwaway LaunchDaemons, then compared
    against `sudo -u` and interactive runs) eventually isolated it to
    /opt/local/bin/python3.10 being completely unsigned
    (`codesign -dvv` reported "code object is not signed at all"),
    while python3.12 happened to be ad-hoc signed. A LaunchDaemon that
    drops privileges via UserName to a non-root, session-less UID gets
    its outbound network silently blocked by macOS if the interpreter
    has no code signature at all -- root and real interactive sessions
    (SSH, sudo -u, Screen Sharing) are unaffected regardless of
    signature, which is what made this so hard to isolate. Along the
    way, TCC (both the system and per-user databases), pf, System
    Extensions, installed configuration profiles, and Screen Time were
    all individually checked and ruled out as the cause.
  - Partial fix: `sudo codesign -f -s - /opt/local/bin/python3.10`
    (ad-hoc signing, same as python3.12 already had). deploy.sh now
    runs this on every deploy regardless of the UserName question
    below, since MacPorts doesn't sign its builds and a future
    `port upgrade`/reinstall of python310 would silently strip the
    signature -- it's a real prerequisite either way.
  - UNRESOLVED: after signing, UserName: maxim still broke outbound
    connectivity on a later deploy. Signing fixed one confirmed,
    reproduced cause (an unsigned interpreter dropped to a non-root,
    session-less UID gets blocked), but evidently isn't sufficient by
    itself -- there's at least one more factor still unidentified.
    com.user.ac.plist is back to running as root (no UserName) as the
    known-working interim state; revisit and find the rest of the
    cause before restoring the privilege drop.

Further small fixes and enhancements, same v0.8, no separate version
bump:

  - requests emitted a RequestsDependencyWarning on every import:
    chardet is an unpinned transitive dependency (via PytorchWildlife)
    that can resolve to a version newer than requests==2.32.3's
    check_compatibility() accepts (>=3.0.2, <6.0.0 -- seen: 7.6.0, well
    outside that range). Fixed by bumping requests to 2.34.2, whose
    check_compatibility() widens the accepted range to <8.0.0 --
    confirmed by reading requests' actual source at that tag, not
    guessed. requests>=2.33 requires Python>=3.10, which this project
    already targets exclusively, so this also meant dropping the
    README's "Python 3.9-3.10" claim down to 3.10 only.
  - Detections were only ever logged as part of a Telegram alert
    (i.e. only once above both the class threshold and, for animals,
    species_threshold); nothing recorded the model's own raw
    confidence for every object it found. Added an INFO log line for
    every raw detection, before threshold filtering, so a probability
    is visible for every object the model reports -- not just ones
    that ended up alerting.
  - MegaDetectorV6's weights are fetched via the `wget` PyPI package,
    which downloads into a `<prefix>.tmp` file in the working
    directory and only renames it to the final name on success. Every
    interrupted AI-model-load during the debugging above (crashes,
    kills, restarts) left one of these orphaned in
    /Users/maxim/nvr/ (WorkingDirectory in com.user.ac.plist) forever,
    since nothing else cleans them up. Added a startup step in
    ai_engine() that removes any matching MDV6*.tmp before loading the
    model.
  - Replaced the separate, self-rotating app log file
    (RotatingFileHandler + [PATHS] log_file + [CLEANUP]
    max_log_size_mb) with logging straight to stdout, which launchd
    already captures to its own file (StandardOutPath). Went from
    three separate output files (stdout, stderr, app log) to two.
    Trade-off: log growth is no longer bounded/rotated in-app: the
    RotatingFileHandler's 1-backup size cap is gone, and nothing
    currently rotates the stdout file it's replaced by either -- worth
    revisiting (e.g. macOS newsyslog) if that file grows unbounded in
    practice.

v0.7 - 2026-09-02
------------------

Security fix, dependency automation, and a first test suite.

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
