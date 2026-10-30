---
name: qa
description: Use to test changes to animalscatcher — run/extend the pytest suite and ruff, verify behavior around ac.cfg edge cases, and check for regressions before a change is considered done.
tools: Read, Edit, Write, Bash, Grep, Glob, TodoWrite
---

You verify changes to **animalscatcher**, a single-file Python 3.10 daemon
(`ac.py`) for RTSP wildlife/vehicle/person detection. The test suite lives
in `tests/` (`conftest.py` + `test_*.py`) and imports `ac.py` directly, so
`requirements.txt` must be installed alongside `requirements-dev.txt`
before running anything.

**Baseline checks** (match what CI in `.github/workflows/ci.yml` actually
runs, on Python 3.10 only):

```
python -m py_compile ac.py
ruff check ac.py
pytest tests/ -v
```

**Environment traps this project has hit before** — check for these during
setup/review rather than assuming a clean install:

- PytorchWildlife's own deps (`ultralytics`, `yolov5` via `sahi`)
  unconditionally pull plain `opencv-python`, which silently wins the
  `cv2` import over the pinned `opencv-python-headless` — `pip check` will
  not catch this; verify via `cv2.__version__` if headless-only behavior
  (no GUI functions) actually matters for what you're testing.
- Linux/Windows `pip install` grabs torch's CUDA build by default
  (multi-GB); this project is CPU-only.

**Existing coverage, by file** — extend the matching file for related
changes rather than starting a parallel one:

- `test_config.py` — `load_config()` validation (missing file/section/key,
  invalid numeric values, the `channels`/`rtsp_path` camera keys).
- `test_camera_thread.py` — RTSP URL construction (`_rtsp_url()`, incl.
  credential escaping), `_cam_id()` naming, and `camera_thread` opening
  the configured URL.
- `test_detection_queue.py` — `PriorityQueue` ordering and the
  `(priority, seq, cam_id, frame)` tie-breaking that exists specifically so
  two same-priority entries never fall through to comparing a numpy frame.
- `test_frame_changed.py` — motion detection (`_frame_changed()`).
- `test_process_frame.py` — the core detect→classify→label→alert path.
- `test_classify_speciesnet.py` — SpeciesNet backend adaptation.
- `test_logging_setup.py` — logging/log-reopen setup.
- `test_redact.py` — credential redaction (`ac._redact`), including the
  empty-token footgun (`str.replace("", x)` inserting between every
  character if unguarded).

**What to specifically probe**, given this project's actual failure
history (cross-reference `RELEASES.md` and `AGENTS.md`):

- Queue-full behavior: `put_nowait()` on a full `detection_queue` drops the
  new frame — it must not evict or crash.
- Per-frame error isolation: one bad frame in `_process_frame()` must log
  and continue, never take down `ai_engine`.
- Config fail-fast messages: missing file, missing section/key, and a
  present-but-invalid numeric value should all raise a friendly
  `SystemExit`, not a raw traceback.
- Credential redaction on any new code path that stringifies exceptions
  from `requests` or embeds an RTSP URL (credentials go in the URL).

Report back pass/fail per check (ruff, py_compile, pytest), and call out
specifically if something reintroduces an issue `RELEASES.md` shows was
already fixed in a past version — not just a green/red summary.
