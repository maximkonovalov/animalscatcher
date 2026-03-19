#!/opt/local/bin/python3

import cv2
import os
import datetime
import time
import threading
import queue
import requests
import configparser
from PytorchWildlife.models import detection as pw_detection

# --- 1. LOAD CONFIGURATION ---
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'lts-mini.cfg')
config.read(config_path)

# Camera & Telegram Settings
USER = config.get('CAMERA', 'user')
PASS = config.get('CAMERA', 'pass')
IP = config.get('CAMERA', 'ip')
PORT = config.get('CAMERA', 'port')
TELEGRAM_TOKEN = config.get('TELEGRAM', 'token')
TELEGRAM_CHAT_ID = config.get('TELEGRAM', 'chat_id')

# Path & Detection Settings
BASE_OUTPUT_FOLDER = config.get('PATHS', 'base_output_folder')
LOG_FILE = config.get('PATHS', 'log_file')
THRESHOLDS = {
    0: config.getfloat('DETECTION', 'threshold_0'), 
    1: config.getfloat('DETECTION', 'threshold_1'), 
    2: config.getfloat('DETECTION', 'threshold_2')
}
COOLDOWN = config.getint('DETECTION', 'cooldown')
FRAME_INTERVAL = config.getint('DETECTION', 'frame_interval')
SUMMARY_INTERVAL = config.getint('DETECTION', 'summary_interval')

# Cleanup Settings
MAX_AGE_DAYS = config.getint('CLEANUP', 'max_age_days')
CLEANUP_INTERVAL = config.getint('CLEANUP', 'cleanup_interval')
MAX_LOG_MB = config.getint('CLEANUP', 'max_log_size_mb')

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;stimeout;5000000"

# --- 2. SHARED DATA & LOCKS ---
detection_queue = queue.Queue(maxsize=15)
stats_lock = threading.Lock()
stats = {
    "Animal": 0, "Person": 0, "Vehicle": 0, 
    "start_time": datetime.datetime.now(), 
    "streams": {} 
}

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except: pass

def send_telegram_photo(photo_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as photo:
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, 
                          files={"photo": photo}, timeout=15)
    except: pass

# --- 3. ENGINE THREADS ---

def cleanup_engine():
    """Removes old snapshots and truncates logs every 24 hours."""
    while True:
        now = time.time()
        cutoff = now - (MAX_AGE_DAYS * 86400)
        deleted_count = 0

        # Snapshot Cleanup
        if os.path.exists(BASE_OUTPUT_FOLDER):
            for cam_dir in os.listdir(BASE_OUTPUT_FOLDER):
                path = os.path.join(BASE_OUTPUT_FOLDER, cam_dir)
                if os.path.isdir(path):
                    for f in os.listdir(path):
                        file_path = os.path.join(path, f)
                        if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff:
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                            except: pass
        
        # Log Truncation
        if os.path.exists(LOG_FILE):
            if (os.path.getsize(LOG_FILE) / (1024 * 1024)) > MAX_LOG_MB:
                with open(LOG_FILE, "w") as f:
                    f.write(f"[{datetime.datetime.now()}] [SYSTEM] Log truncated (Exceeded {MAX_LOG_MB}MB)\n")

        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.datetime.now()}] [SYSTEM] Cleanup: Removed {deleted_count} old snapshots.\n")
        
        time.sleep(CLEANUP_INTERVAL * 3600)

def summary_engine():
    """Periodically sends a summary of detections and stream stats."""
    while True:
        time.sleep(SUMMARY_INTERVAL * 3600)
        with stats_lock:
            now = datetime.datetime.now()
            s_info = "\n".join([f"- {k}: {v['status']} ({v['res']})" for k,v in stats["streams"].items()])
            report = (f"--- NVR SUMMARY ---\n"
                      f"Range: {stats['start_time'].strftime('%H:%M')} - {now.strftime('%H:%M')}\n\n"
                      f"STREAMS:\n{s_info}\n\n"
                      f"DETECTIONS:\n- Animals: {stats['Animal']}\n- People: {stats['Person']}\n- Vehicles: {stats['Vehicle']}")
            # Reset counters
            stats.update({"Animal": 0, "Person": 0, "Vehicle": 0, "start_time": now})
        send_telegram_message(report)

def camera_thread(cam_num):
    """Maintains RTSP connection and samples frames for the AI."""
    cam_id = f"cam0{cam_num}"
    rtsp_url = f"rtsp://{USER}:{PASS}@{IP}:{PORT}/Streaming/Channels/{cam_num}02"
    os.makedirs(os.path.join(BASE_OUTPUT_FOLDER, cam_id), exist_ok=True)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    f_idx = 0
    while True:
        success, frame = cap.read()
        with stats_lock:
            if not success or frame is None:
                stats["streams"][cam_id] = {"status": "OFFLINE", "res": "N/A"}
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                continue
            stats["streams"][cam_id] = {"status": "ONLINE", "res": f"{int(cap.get(3))}x{int(cap.get(4))}"}

        if f_idx % FRAME_INTERVAL == 0 and not detection_queue.full():
            detection_queue.put((cam_id, frame))
        f_idx += 1

def ai_engine():
    """Processes frames from the queue using MegaDetectorV6."""
    model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c", device="cpu", pretrained=True)
    last_det = {}; motion_val = {}; names = {0: "Animal", 1: "Person", 2: "Vehicle"}
    send_telegram_message("NVR SYSTEM ONLINE (DAEMON MODE)")

    while True:
        cam_id, frame = detection_queue.get()
        results = model.single_image_detection(frame)
        det = results.get("detections")
        seen = {0: False, 1: False, 2: False}

        if det is not None and len(det.confidence) > 0:
            for i in range(len(det.confidence)):
                conf, cls = float(det.confidence[i]), int(det.class_id[i])
                if conf > THRESHOLDS.get(cls, 0.5):
                    seen[cls] = True
                    # Basic motion validation (must be seen in consecutive sampled frames)
                    if motion_val.get((cam_id, cls), False) and (time.time() - last_det.get((cam_id, cls), 0) > COOLDOWN):
                        label = names[cls]
                        with stats_lock: stats[label] += 1
                        
                        fname = f"{cam_id}_{int(time.time())}.jpg"
                        fpath = os.path.join(BASE_OUTPUT_FOLDER, cam_id, fname)
                        cv2.imwrite(fpath, frame)
                        
                        caption = f"ALERT: {label} detected on {cam_id} (Conf: {conf:.2f})"
                        threading.Thread(target=send_telegram_photo, args=(fpath, caption)).start()
                        last_det[(cam_id, cls)] = time.time()

        for c in [0,1,2]: motion_val[(cam_id, c)] = seen[c]
        detection_queue.task_done()

# --- 4. STARTUP ---
if __name__ == "__main__":
    # Launch background workers
    for t in [ai_engine, summary_engine, cleanup_engine]:
        threading.Thread(target=t, daemon=True).start()
    
    # Launch camera streams (Cam 4, 5, 6)
    for n in [4, 5, 6]:
        threading.Thread(target=camera_thread, args=(n,), daemon=True).start()
        time.sleep(2)
        
    while True:
        time.sleep(1)

# EOF
