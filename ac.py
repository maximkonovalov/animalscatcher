#!/opt/local/bin/python3

import cv2
import os
import datetime
import time
import threading
import queue
import requests
import configparser
from urllib.parse import quote
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

# --- 0. VERSIONING ---
VERSION = "0.5"

# --- 1. LOAD CONFIGURATION ---
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'ac.cfg')
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

# Species Settings
SPECIES_THRESHOLD = 0.45
# Tolerance for "Static" object detection (approx 3% of frame width/height)
STATIC_TOLERANCE = 0.03

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ("rtsp_transport;tcp|"
                                               "stimeout;5000000")

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
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except: pass

def send_telegram_photo(photo_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    try:
        with open(photo_path, "rb") as photo:
            requests.post(url, data=payload, files={"photo": photo}, timeout=15)
    except: pass

# --- 3. ENGINE THREADS ---

def cleanup_engine():
    """Removes old snapshots and truncates logs every 24 hours."""
    while True:
        now = time.time()
        cutoff = now - (MAX_AGE_DAYS * 86400)
        deleted_count = 0
        if os.path.exists(BASE_OUTPUT_FOLDER):
            for cam_dir in os.listdir(BASE_OUTPUT_FOLDER):
                path = os.path.join(BASE_OUTPUT_FOLDER, cam_dir)
                if os.path.isdir(path):
                    for f in os.listdir(path):
                        f_path = os.path.join(path, f)
                        if (os.path.isfile(f_path) and
                            os.path.getmtime(f_path) < cutoff):
                            try:
                                os.remove(f_path)
                                deleted_count += 1
                            except: pass
        if os.path.exists(LOG_FILE):
            if (os.path.getsize(LOG_FILE) / (1024 * 1024)) > MAX_LOG_MB:
                with open(LOG_FILE, "w") as f:
                    f.write(f"[{datetime.datetime.now()}] [SYSTEM] "
                            f"Log truncated (Exceeded {MAX_LOG_MB}MB)\n")
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.datetime.now()}] [SYSTEM] "
                    f"Cleanup: Removed {deleted_count} old snapshots.\n")
        time.sleep(CLEANUP_INTERVAL * 3600)

def summary_engine():
    """Periodically sends a summary of detections and stream stats."""
    while True:
        time.sleep(SUMMARY_INTERVAL * 3600)
        with stats_lock:
            now = datetime.datetime.now()
            s_list = [f"- {k}: {v['status']} ({v['res']})"
                      for k,v in stats["streams"].items()]
            s_info = "\n".join(s_list)
            report = (f"--- NVR SUMMARY ---\n"
                      f"Version: {VERSION}\n"
                      f"Range: {stats['start_time'].strftime('%H:%M')} - "
                      f"{now.strftime('%H:%M')}\n\n"
                      f"STREAMS:\n{s_info}\n\n"
                      f"DETECTIONS:\n- Animals: {stats['Animal']}\n"
                      f"- People: {stats['Person']}\n"
                      f"- Vehicles: {stats['Vehicle']}")
            stats.update({"Animal": 0, "Person": 0, "Vehicle": 0,
                          "start_time": now})
        send_telegram_message(report)

def camera_thread(cam_num):
    """Maintains RTSP connection and samples frames for the AI."""
    cam_id = f"cam0{cam_num}"
    user = quote(USER, safe='')
    password = quote(PASS, safe='')
    url = f"rtsp://{user}:{password}@{IP}:{PORT}/Streaming/Channels/{cam_num}02"
    os.makedirs(os.path.join(BASE_OUTPUT_FOLDER, cam_id), exist_ok=True)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    f_idx = 0
    while True:
        success, frame = cap.read()
        with stats_lock:
            if not success or frame is None:
                stats["streams"][cam_id] = {"status": "OFFLINE", "res": "N/A"}
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                continue
            stats["streams"][cam_id] = {"status": "ONLINE",
                                        "res": f"{int(cap.get(3))}x"
                                               f"{int(cap.get(4))}"}
        if f_idx % FRAME_INTERVAL == 0 and not detection_queue.full():
            detection_queue.put((cam_id, frame))
        f_idx += 1

def ai_engine():
    """Processes frames: Detects objects and filters static false positives."""
    detector = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c",
                                           device="cpu", pretrained=True)
    classifier = pw_classification.DeepfauneClassifier(device="cpu")
    last_det = {}; motion_val = {}; last_box = {}
    names = {0: "Animal", 1: "Person", 2: "Vehicle"}
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
    send_telegram_message(f"The animal catcher is online, version {VERSION}")
    while True:
        cam_id, frame = detection_queue.get()
        results = detector.single_image_detection(frame)
        det = results.get("detections")
        seen = {0: False, 1: False, 2: False}
        if det is not None and len(det.confidence) > 0:
            h, w, _ = frame.shape
            for i in range(len(det.confidence)):
                conf, cls = float(det.confidence[i]), int(det.class_id[i])
                if conf > THRESHOLDS.get(cls, 0.5):
                    seen[cls] = True
                    obj_name = names.get(cls, "Object")
                    label = f"{obj_name} ({conf:.2f})"
                    box = det.xyxy[i]
                    x1, y1, x2, y2 = (int(box[0]), int(box[1]),
                                      int(box[2]), int(box[3]))
                    if cls == 0 and conf > SPECIES_THRESHOLD:
                        crop = frame[max(0, y1):min(h, y2),
                                     max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            try:
                                s_res = classifier.single_image_classification(crop)
                                res = s_res[0] if isinstance(s_res, list) else s_res
                                s_label = (res.get('label') or
                                           res.get('prediction') or
                                           res.get('y_pred') or "Unknown")
                                s_conf = next((v for v in
                                    (res.get('confidence'), res.get('y_conf'))
                                    if v is not None), 0.0)
                                if s_conf > SPECIES_THRESHOLD:
                                    label = f"{obj_name}: {s_label} ({s_conf:.2f})"
                            except: pass
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  colors.get(cls, (0, 255, 0)), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colors.get(cls, (0, 255, 0)), 2)
                    d_key = (cam_id, cls)
                    if (motion_val.get(d_key, False) and
                        (time.time() - last_det.get(d_key, 0) > COOLDOWN)):
                        is_static = False
                        if d_key in last_box:
                            lb = last_box[d_key]
                            if (abs(x1-lb[0]) < w*STATIC_TOLERANCE and
                                abs(y1-lb[1]) < h*STATIC_TOLERANCE):
                                is_static = True
                        if not is_static:
                            with stats_lock: stats[obj_name] += 1
                            last_box[d_key] = (x1, y1, x2, y2)
                            fname = f"{cam_id}_{int(time.time())}.jpg"
                            fpath = os.path.join(BASE_OUTPUT_FOLDER,
                                                 cam_id, fname)
                            cv2.imwrite(fpath, frame)
                            threading.Thread(target=send_telegram_photo,
                                args=(fpath, f"ALERT: {label} on {cam_id}")).start()
                            last_det[d_key] = time.time()
                        else:
                            with open(LOG_FILE, "a") as f:
                                f.write(f"[{datetime.datetime.now()}] "
                                        f"[FILTER] Static {obj_name} ignored "
                                        f"on {cam_id} at ({x1},{y1})\n")
        for c in [0,1,2]: motion_val[(cam_id, c)] = seen[c]
        detection_queue.task_done()

# --- 4. STARTUP ---
if __name__ == "__main__":
    print(f"Starting Animal Catcher v{VERSION}...")
    for t in [ai_engine, summary_engine, cleanup_engine]:
        threading.Thread(target=t, daemon=True).start()
    for n in [4, 5, 6]:
        threading.Thread(target=camera_thread, args=(n,), daemon=True).start()
        time.sleep(2)
    while True: time.sleep(1)

# EOF
