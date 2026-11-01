# System Architecture & Agents

**animalscatcher** operates using a multi-threaded architecture to ensure
real-time processing of RTSP streams without blocking the AI engine.

## 1. Agent Overview

The system consists of four primary agent types running in parallel:

| Agent | Responsibility | Frequency |
| :--- | :--- | :--- |
| **Camera Agent** | Maintains the RTSP connection and watches it for motion. | Continuous |
| **AI Agent** | Performs detection and species classification. | Motion-driven |
| **Summary Agent** | Compiles stats and sends periodic reports. | Configurable |
| **Cleanup Agent** | Purges old snapshots to manage storage. | Configurable |

---

## 2. Detailed Workflows

### Camera Agent (`camera_thread`)
Each channel listed in `[CAMERA] channels` spawns its own dedicated thread.
* **Stream URL:** `_rtsp_url()` builds it from `ip`, `port`, the
  (percent-encoded) login, and `rtsp_path` with `{channel}` filled in;
  `_cam_id()` names the snapshot folder and thread (`cam04`, ...). The
  URL embeds the credentials, so it goes straight to `cv2.VideoCapture`
  and is never logged. See README's Camera Setup section.
* **Resilience:** If a stream drops, the agent enters a retry loop.
* **Motion detection:** Each camera runs its own `MotionDetector` (the
  tunings are the `[DETECTION]` motion keys, the same for every camera),
  so whether a camera is watched never depends on another. Every
  `motion_check_interval`-th raw
  frame is compared with a reference frame (the previous check, or the one
  `motion_lookback` seconds earlier): shrunk to 160x120, gray, blurred,
  `motion_pixel_delta` per pixel, and the largest connected patch of
  changed pixels must cover more than `motion_threshold` of the frame. Only
  a frame that shows motion -- the later of the two compared -- goes on to
  the AI; nothing is sampled on a timer, so a camera with no motion costs
  the AI nothing. `grab()` skips the BGR conversion on frames that aren't
  checked, because the camera threads measurably compete with `ai_engine`
  for CPU on the target hardware.
* **Motion log:** The first check of a burst of motion -- no motion on that
  camera for `MOTION_LOG_GAP` (10 s) before it -- is logged as
  `[MOTION] cam04: 0.350% of the frame changed (threshold 0.120%)`. The
  rest of the burst is counted (the summary's "motion") but not logged, so
  the gap runs from the latest motion, not from where the burst began.
* **Hand-off:** `detection_queue` (`FrameQueue`) is the one queue of the
  frames that show motion, shared by every camera: `AI_QUEUE_SIZE` (15)
  frames, oldest first, which is what absorbs a spike of motion. When it is
  full, `offer()` drops the oldest entry to make room for the new frame and
  returns the camera that entry came from; `camera_thread` counts the drop
  against that camera ("dropped" in the summary). `ai_engine` takes frames
  with `take()`, oldest first, whatever camera they came from. That is the
  whole algorithm, on purpose: nothing per camera, no priorities, no
  scheduling. What it costs is that the cameras share the room: a camera in
  constant motion fills the queue, pushes the others' older frames out and
  gets most of the AI's passes (the calibration visits in README's Motion
  Detection section show how much), and the per-camera drop counts are how
  that would show up.

### AI Inference Agent (`ai_engine`)
The core "brain" of the system. Before loading any model, it clears out
any `MDV6*.tmp` partial download left behind in the working directory by
a previous, interrupted weights download (MegaDetectorV6 fetches its
weights via the `wget` package, which doesn't clean these up itself).
It then takes frames from `detection_queue.take()`, oldest first whatever
camera they came from (see Camera Agent above); each frame is handed to
`_process_frame()`, which does the actual detection, classification, and
alerting.
1. **Detection:** Uses **MegaDetectorV6** for Animals and People. The
   model also reports vehicles, but that class is dropped outright (a class
   without a `threshold_N`, so not in `THRESHOLDS`): no log line, no count,
   no alert -- and no vehicle threshold in `ac.cfg`, though one left over
   from earlier versions is tolerated with a startup warning. Every raw
   Animal or Person detection is logged with its confidence (before
   threshold filtering), independent of whether it ends up alerting. The
   model still has to look at a frame to find that out, so ignoring
   vehicles does not make the AI any cheaper.
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
4. **Alerting:** If the sighting is confirmed by the previous look at the
   same camera and class (at most `confirm_window` seconds earlier --
   `_expire_stale_latch()` enforces that, since frames now only follow
   motion and the previous one can be hours old), `cooldown` has passed,
   and the detection isn't static (within `static_tolerance`, also
   config-driven), it triggers an asynchronous Telegram upload.
5. **Isolation:** A per-frame error is caught, logged, and skipped rather
   than killing the agent — one bad frame doesn't take down detection.

### Summary Agent (`summary_engine`)
A background observer that tracks stream health and detection counts. It
provides a heartbeat to ensure the system is active. Also reports, per
camera, how many frames were checked for motion, how many showed it, how
many `ai_engine` analysed (and how many of those had an animal on them --
`_process_frame` counts that whether or not an alert follows), how many
of its frames were dropped from the queue, the strongest change seen
beside the motion threshold, and how long frames waited in the queue for
the AI. An AI QUEUE block under the cameras adds them up: frames offered
to the queue and how many were dropped, with their share. A dropped frame
is the oldest one waiting when a newer one reached a full queue, so it is
counted against the camera it came from. Drops are expected in any long
burst (a camera with motion offers four frames a second, the AI takes about
one in 3.5 s), so they measure how much more motion there was than the AI
could take, not how many events it missed. All cameras share one
`ai_engine` thread, so this is both the visibility into whether it keeps up
and the data for tuning the motion settings (see README's Motion Detection
section).

### Cleanup Agent (`cleanup_engine`)
A maintenance worker that keeps the host system stable.
* **Storage:** Deletes snapshots older than `max_age_days`.
* **Logs:** Not its concern — the daemon logs to stdout/stderr rather
  than a file it owns, so it has nothing to rotate itself. Rotation of
  those files is handled externally (macOS newsyslog, via
  assets/com.user.ac.newsyslog.conf), coordinated with a SIGUSR1 handler in
  `__main__` (`_handle_log_reopen`) that reopens fresh file
  descriptors after each rotation. See README's Log Rotation section.

---

## 3. Data Flow Diagram

1. **RTSP Stream** -> **Camera Agent** (motion check, per camera)
2. **Camera Agent** -> **`detection_queue`** (frames that show motion; one
   queue for all cameras, the oldest frame is dropped and counted when it
   is full)
3. **`detection_queue`** -> **AI Agent** (Inference & Labeling)
4. **AI Agent** -> **Telegram API** (User Notification)
5. **AI Agent** -> **Local Storage** (Snapshot persistence)

---

## 4. Design Philosophy
* **Asynchronous Notifications:** Telegram photos are sent via a bounded
  `ThreadPoolExecutor` (4 workers) to prevent blocking the AI pipeline
  without spawning an unbounded thread per detection.
* **A simple hand-off:** Each camera judges its own motion; a frame that
  shows it goes into one bounded queue shared by all cameras, the oldest
  frame is dropped when the queue is full and the drops are counted and
  reported; one AI thread takes frames oldest first. Nothing more: the
  hand-off was a shared priority queue that dropped new frames (v0.15), then
  a newest-frame slot per camera that left nothing to absorb a spike
  (v0.16), and a per-camera queue served in turn was built and set aside
  as more mechanism than the problem needs. Keep it this simple unless
  there is evidence that it costs something worth a mechanism -- the
  per-camera drop counts in the summary are that evidence.
* **Thread Safety:** A `stats_lock` protects the global `stats` dictionary.
* **Resilience:** Each agent's main loop is wrapped in `try/except`, so an
  unexpected error is logged and the loop continues rather than silently
  killing that agent while the rest of the process keeps running.
* **Config Validation:** `load_config()` fails fast with a friendly message
  (missing file, missing key, or an invalid value) instead of letting a
  bad `ac.cfg` surface as a raw traceback with no log record.
* **Graceful Shutdown:** `SIGTERM`/`SIGINT` are caught to log the shutdown
  and cancel any queued-but-unstarted Telegram uploads
  (`photo_executor.shutdown(wait=False, cancel_futures=True)`) so a
  backlog doesn't hold up exit -- though an upload already in flight can
  still delay it briefly.
* **Log Reopening:** `SIGUSR1` is caught (`_handle_log_reopen`) to
  `dup2()` fresh file descriptors onto stdout/stderr after an external
  tool (newsyslog) rotates the files launchd redirects them to --
  reopening in place rather than restarting the process. See README's
  Log Rotation section for the full mechanism.
* **Style:** Minimalist, efficient, and robust against network interruptions.

---
#EOF
