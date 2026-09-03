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
| **Cleanup Agent** | Manages storage and log rotation. | Configurable |

---

## 2. Detailed Workflows

### Camera Agent (`camera_thread`)
Each camera defined in the startup sequence spawns its own dedicated thread.
* **Resilience:** If a stream drops, the agent enters a retry loop.
* **Sampling:** To save CPU, only every $N$-th frame (via `frame_interval`)
  is sent to the shared `detection_queue`.

### AI Inference Agent (`ai_engine`)
The core "brain" of the system. It monitors the `detection_queue` and
processes frames using a First-In-First-Out (FIFO) logic; each frame is
handed to `_process_frame()`, which does the actual detection,
classification, and alerting.
1. **Detection:** Uses **MegaDetectorV6** for Animals, People, or Vehicles.
2. **Classification:** If an animal is detected above `species_threshold`
   (config-driven, `ac.cfg` `[DETECTION]`), the agent crops the area and
   passes it to **DeepFaune** for species identification.
3. **Labeling:** Annotates the frame with type, species, and confidence.
4. **Alerting:** If motion is confirmed (past `cooldown`) and the
   detection isn't static (within `static_tolerance`, also config-driven),
   it triggers an asynchronous Telegram upload.
5. **Isolation:** A per-frame error is caught, logged, and skipped rather
   than killing the agent — one bad frame doesn't take down detection.

### Summary Agent (`summary_engine`)
A background observer that tracks stream health and detection counts. It
provides a heartbeat to ensure the system is active.

### Cleanup Agent (`cleanup_engine`)
A maintenance worker that keeps the host system stable.
* **Storage:** Deletes snapshots older than `max_age_days`.
* **Logs:** Rotation is handled separately by a `RotatingFileHandler`
  on the shared `logger`, which rolls over automatically once the log
  exceeds `max_log_size_mb` — the Cleanup Agent itself only manages
  snapshots.

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
* **Style:** Minimalist, efficient, and robust against network interruptions.

---
#EOF
