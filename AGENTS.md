# System Architecture & Agents

**animalscatcher** operates using a multi-threaded architecture to ensure
real-time processing of RTSP streams without blocking the AI engine.

## 1. Agent Overview

The system consists of four primary agent types running in parallel:

| Agent | Responsibility | Frequency |
| :--- | :--- | :--- |
| **Camera Agent** | Maintains RTSP connection and samples frames. | Continuous |
| **AI Agent** | Performs detection and species classification. | Queue-driven |
| **Summary Agent** | Compiles stats and sends periodic reports. | Configurable |
| **Cleanup Agent** | Purges old snapshots to manage storage. | Configurable |

---

## 2. Detailed Workflows

### Camera Agent (`camera_thread`)
Each camera defined in the startup sequence spawns its own dedicated thread.
* **Resilience:** If a stream drops, the agent enters a retry loop.
* **Sampling:** To save CPU, only every $N$-th frame (via `frame_interval`)
  is sent to the shared `detection_queue` -- *plus* any frame where
  `_frame_changed()` detects motion (via `motion_threshold`), checked
  on every raw frame since it's far cheaper than the AI pipeline. This
  is additive, not a replacement: `frame_interval` alone can miss a
  fast animal's entire visible window between samples, so motion
  detection adds responsiveness on top of the existing periodic
  baseline rather than replacing it.
* **Queue priority:** `detection_queue` is a `PriorityQueue`, not FIFO
  -- motion-triggered frames (priority 0) are drained by the AI Agent
  before frame_interval-only ones (priority 1) waiting alongside them,
  via a `(priority, seq, cam_id, frame)` tuple where `seq` is a
  strictly increasing tie-breaker (needed so same-priority comparisons
  never fall through to comparing `frame`, a numpy array, which raises).
  This only affects drain order for frames already queued, though --
  `put_nowait()` on an already-full queue still drops the new frame
  regardless of its priority; it doesn't evict a lower-priority one to
  make room.

### AI Inference Agent (`ai_engine`)
The core "brain" of the system. Before loading any model, it clears out
any `MDV6*.tmp` partial download left behind in the working directory by
a previous, interrupted weights download (MegaDetectorV6 fetches its
weights via the `wget` package, which doesn't clean these up itself).
It then monitors the `detection_queue` and processes frames using a
First-In-First-Out (FIFO) logic; each frame is handed to
`_process_frame()`, which does the actual detection, classification,
and alerting.
1. **Detection:** Uses **MegaDetectorV6** for Animals, People, or
   Vehicles. Every raw detection is logged with its confidence (before
   threshold filtering), independent of whether it ends up alerting.
2. **Classification:** If an animal is detected above `species_threshold`
   (config-driven, `ac.cfg` `[DETECTION]`), the agent crops the area --
   padded beyond the raw detection box by `crop_padding` (a fraction of
   the box's own size, default 15%), so a tight box doesn't clip a
   tail, ear, or antler the classifier needs -- and passes it to a
   configurable classifier (`[DETECTION] classifier`):
   **DFNE** by default ("Deepfaune-New-England", a USGS retrain of
   Deepfaune on North American species), or **SpeciesNet** (Google's
   2498-taxa classifier) for regions DFNE covers poorly. `_classify_dfne`
   and `_classify_speciesnet` adapt each backend's own API to the same
   `(label, confidence)` pair `_process_frame()` actually uses -- see
   README's Species Classifier section.
3. **Labeling:** Annotates the frame with type, species, and confidence.
4. **Alerting:** If motion is confirmed (past `cooldown`) and the
   detection isn't static (within `static_tolerance`, also config-driven),
   it triggers an asynchronous Telegram upload.
5. **Isolation:** A per-frame error is caught, logged, and skipped rather
   than killing the agent — one bad frame doesn't take down detection.

### Summary Agent (`summary_engine`)
A background observer that tracks stream health and detection counts. It
provides a heartbeat to ensure the system is active. Also reports how
many sampled frames were dropped because `detection_queue` was full --
all cameras share one `ai_engine` thread, and MegaDetector's per-frame
CPU cost means sampling can outpace processing; this is the visibility
into that, rather than it failing silently.

### Cleanup Agent (`cleanup_engine`)
A maintenance worker that keeps the host system stable.
* **Storage:** Deletes snapshots older than `max_age_days`.
* **Logs:** Not its concern — the daemon logs to stdout/stderr rather
  than a file it owns, so it has nothing to rotate itself. Rotation of
  those files is handled externally (macOS newsyslog, via
  com.user.ac.newsyslog.conf), coordinated with a SIGUSR1 handler in
  `__main__` (`_handle_log_reopen`) that reopens fresh file
  descriptors after each rotation. See README's Log Rotation section.

---

## 3. Data Flow Diagram

1. **RTSP Stream** -> **Camera Agent** (Frame Sampling)
2. **Camera Agent** -> **Shared Queue** (Thread-safe buffer)
3. **Shared Queue** -> **AI Agent** (Inference & Labeling)
4. **AI Agent** -> **Telegram API** (User Notification)
5. **AI Agent** -> **Local Storage** (Snapshot persistence)

---

## 4. Design Philosophy
* **Asynchronous Notifications:** Telegram photos are sent via a bounded
  `ThreadPoolExecutor` (4 workers) to prevent blocking the AI pipeline
  without spawning an unbounded thread per detection.
* **Thread Safety:** A `stats_lock` protects the global `stats` dictionary.
* **Resilience:** Each agent's main loop is wrapped in `try/except`, so an
  unexpected error is logged and the loop continues rather than silently
  killing that agent while the rest of the process keeps running.
* **Config Validation:** `load_config()` fails fast with a friendly message
  (missing file, missing key, or an invalid value) instead of letting a
  bad `ac.cfg` surface as a raw traceback with no log record.
* **Graceful Shutdown:** `SIGTERM`/`SIGINT` are caught to log the shutdown
  and drain `photo_executor` before exiting, rather than being killed
  mid-write.
* **Log Reopening:** `SIGUSR1` is caught (`_handle_log_reopen`) to
  `dup2()` fresh file descriptors onto stdout/stderr after an external
  tool (newsyslog) rotates the files launchd redirects them to --
  reopening in place rather than restarting the process. See README's
  Log Rotation section for the full mechanism.
* **Style:** Minimalist, efficient, and robust against network interruptions.

---
#EOF
