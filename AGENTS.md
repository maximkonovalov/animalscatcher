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
| **Cleanup Agent** | Manages storage and log rotation. | 24 Hours |

---

## 2. Detailed Workflows

### Camera Agent (`camera_thread`)
Each camera defined in the startup sequence spawns its own dedicated thread.
* **Resilience:** If a stream drops, the agent enters a retry loop.
* **Sampling:** To save CPU, only every $N$-th frame (via `frame_interval`)
  is sent to the shared `detection_queue`.

### AI Inference Agent (`ai_engine`)
The core "brain" of the system. It monitors the `detection_queue` and
processes frames using a First-In-First-Out (FIFO) logic.
1. **Detection:** Uses **MegaDetectorV6** for Animals, People, or Vehicles.
2. **Classification:** If an animal is detected, the agent crops the area
   and passes it to **DeepFaune** for species identification.
3. **Labeling:** Annotates the frame with type, species, and confidence.
4. **Alerting:** If motion is confirmed (past `cooldown`), it triggers
   an asynchronous Telegram upload.

### Summary Agent (`summary_engine`)
A background observer that tracks stream health and detection counts. It
provides a heartbeat to ensure the system is active.

### Cleanup Agent (`cleanup_engine`)
A maintenance worker that keeps the host system stable.
* **Storage:** Deletes snapshots older than `max_age_days`.
* **Logs:** Truncates `ac.log` if it exceeds `max_log_size_mb`.

---

## 3. Data Flow Diagram

1. **RTSP Stream** -> **Camera Agent** (Frame Sampling)
2. **Camera Agent** -> **Shared Queue** (Thread-safe buffer)
3. **Shared Queue** -> **AI Agent** (Inference & Labeling)
4. **AI Agent** -> **Telegram API** (User Notification)
5. **AI Agent** -> **Local Storage** (Snapshot persistence)

---

## 4. Design Philosophy
* **Asynchronous Notifications:** Telegram photos are sent in "fire-and-forget"
  threads to prevent blocking the AI pipeline.
* **Thread Safety:** A `stats_lock` protects the global `stats` dictionary.
* **Style:** Minimalist, efficient, and robust against network interruptions.

---
#EOF
