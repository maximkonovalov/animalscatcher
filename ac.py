#!/usr/bin/env python3

import configparser
import datetime
import logging
import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from urllib.parse import quote

import cv2
import requests
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.models import detection as pw_detection

# --- 0. VERSIONING ---
VERSION = "0.8"

# RTSP channel numbers to monitor (see camera_thread / STARTUP).
CAMERA_CHANNELS = [4, 5, 6]

# --- 1. LOAD CONFIGURATION ---
def load_config(path):
    """Parse and validate ac.cfg, returning a dict of typed settings.
    Raises SystemExit with a friendly message if the file is missing,
    unreadable, or a required key is missing/invalid -- covers both
    configparser errors (missing section/key) and getfloat/getint
    ValueErrors (a key present but not a valid number)."""
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise SystemExit(f"Config file not found or unreadable: {path}")
    try:
        return {
            'user': parser.get('CAMERA', 'user'),
            'pass': parser.get('CAMERA', 'pass'),
            'ip': parser.get('CAMERA', 'ip'),
            'port': parser.get('CAMERA', 'port'),
            'telegram_token': parser.get('TELEGRAM', 'token'),
            'telegram_chat_id': parser.get('TELEGRAM', 'chat_id'),
            'base_output_folder': parser.get('PATHS', 'base_output_folder'),
            'log_file': parser.get('PATHS', 'log_file'),
            'thresholds': {
                0: parser.getfloat('DETECTION', 'threshold_0'),
                1: parser.getfloat('DETECTION', 'threshold_1'),
                2: parser.getfloat('DETECTION', 'threshold_2'),
            },
            'cooldown': parser.getint('DETECTION', 'cooldown'),
            'frame_interval': parser.getint('DETECTION', 'frame_interval'),
            'summary_interval': parser.getint('DETECTION', 'summary_interval'),
            'species_threshold': parser.getfloat('DETECTION',
                                                  'species_threshold'),
            'static_tolerance': parser.getfloat('DETECTION',
                                                'static_tolerance'),
            'max_age_days': parser.getint('CLEANUP', 'max_age_days'),
            'cleanup_interval': parser.getint('CLEANUP', 'cleanup_interval'),
            'max_log_size_mb': parser.getint('CLEANUP', 'max_log_size_mb'),
        }
    except (configparser.Error, ValueError) as e:
        raise SystemExit(f"Invalid or incomplete {path}: {e} "
                         f"(see ac.cfg.example for the required keys)")

config_path = os.environ.get('AC_CONFIG_PATH') or os.path.join(
    os.path.dirname(__file__), 'ac.cfg')
_cfg = load_config(config_path)

# Camera & Telegram Settings
USER = _cfg['user']
PASS = _cfg['pass']
IP = _cfg['ip']
PORT = _cfg['port']
TELEGRAM_TOKEN = _cfg['telegram_token']
TELEGRAM_CHAT_ID = _cfg['telegram_chat_id']

# Path & Detection Settings
BASE_OUTPUT_FOLDER = _cfg['base_output_folder']
LOG_FILE = _cfg['log_file']
THRESHOLDS = _cfg['thresholds']
COOLDOWN = _cfg['cooldown']
FRAME_INTERVAL = _cfg['frame_interval']
SUMMARY_INTERVAL = _cfg['summary_interval']
SPECIES_THRESHOLD = _cfg['species_threshold']
# Tolerance for "Static" detection, as a fraction of frame width/height.
STATIC_TOLERANCE = _cfg['static_tolerance']

# Cleanup Settings
MAX_AGE_DAYS = _cfg['max_age_days']
CLEANUP_INTERVAL = _cfg['cleanup_interval']
MAX_LOG_MB = _cfg['max_log_size_mb']

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ("rtsp_transport;tcp|"
                                               "stimeout;5000000")

# --- 2. LOGGING ---
# Thread-safe, self-rotating log (replaces manual open/append/truncate).
logger = logging.getLogger("animalcatcher")
logger.setLevel(logging.INFO)
_log_handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_MB * 1024 * 1024,
                                   backupCount=1)
_log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
logger.addHandler(_log_handler)

# --- 3. SHARED DATA & LOCKS ---
detection_queue = queue.Queue(maxsize=15)
stats_lock = threading.Lock()
stats = {
    "Animal": 0, "Person": 0, "Vehicle": 0,
    "start_time": datetime.datetime.now(),
    "streams": {}
}
# Bounds concurrent Telegram photo uploads instead of spawning one thread
# per detection.
photo_executor = ThreadPoolExecutor(max_workers=4,
                                    thread_name_prefix="telegram-upload")

def _redact(text):
    """Strip the Telegram bot token out of a string before logging it --
    request exceptions often stringify the full request URL, which
    embeds the token."""
    if TELEGRAM_TOKEN:
        text = text.replace(TELEGRAM_TOKEN, "<redacted>")
    return text

def send_telegram_message(message):
    """Best-effort notification: never raises, so callers don't need to
    guard against a Telegram/network hiccup taking down their thread."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        logger.warning(f"[TELEGRAM] Failed to send message: {_redact(str(e))}")

def send_telegram_photo(photo_path, caption):
    """Best-effort notification: never raises, same contract as
    send_telegram_message."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    try:
        with open(photo_path, "rb") as photo:
            requests.post(url, data=payload, files={"photo": photo}, timeout=15)
    except Exception as e:
        logger.warning(f"[TELEGRAM] Failed to send photo {photo_path}: "
                       f"{_redact(str(e))}")

# --- 4. ENGINE THREADS ---

def cleanup_engine():
    """Removes old snapshots every 24 hours. Log rotation is handled by
    the RotatingFileHandler on `logger`."""
    while True:
        try:
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
                                except OSError as e:
                                    logger.warning(f"[SYSTEM] Failed to "
                                                   f"remove {f_path}: {e}")
            logger.info(f"[SYSTEM] Cleanup: Removed {deleted_count} old "
                       f"snapshots.")
        except Exception as e:
            logger.error(f"[SYSTEM] cleanup_engine iteration failed: {e}")
        time.sleep(CLEANUP_INTERVAL * 3600)

def summary_engine():
    """Periodically sends a summary of detections and stream stats."""
    while True:
        time.sleep(SUMMARY_INTERVAL * 3600)
        try:
            with stats_lock:
                now = datetime.datetime.now()
                s_list = [f"- {k}: {v['status']} ({v['res']})"
                          for k, v in stats["streams"].items()]
                s_info = "\n".join(s_list)
                report = (f"--- NVR SUMMARY ---\n"
                          f"Version: {VERSION}\n"
                          f"Range: {stats['start_time'].strftime('%d/%m/%Y %H:%M')} - "
                          f"{now.strftime('%d/%m/%Y %H:%M')}\n\n"
                          f"STREAMS:\n{s_info}\n\n"
                          f"DETECTIONS:\n- Animals: {stats['Animal']}\n"
                          f"- People: {stats['Person']}\n"
                          f"- Vehicles: {stats['Vehicle']}")
                stats.update({"Animal": 0, "Person": 0, "Vehicle": 0,
                              "start_time": now})
            send_telegram_message(report)
        except Exception as e:
            logger.error(f"[SYSTEM] summary_engine iteration failed: {e}")

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
        try:
            success, frame = cap.read()
            if not success or frame is None:
                with stats_lock:
                    stats["streams"][cam_id] = {"status": "OFFLINE",
                                                "res": "N/A"}
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            with stats_lock:
                stats["streams"][cam_id] = {"status": "ONLINE",
                                            "res": f"{w}x{h}"}
            if f_idx % FRAME_INTERVAL == 0:
                try:
                    detection_queue.put_nowait((cam_id, frame))
                except queue.Full:
                    pass
            f_idx += 1
        except Exception as e:
            logger.error(f"[SYSTEM] camera_thread({cam_id}) iteration "
                        f"failed: {e}")
            time.sleep(5)

def _process_frame(cam_id, frame, detector, classifier, names, colors,
                   last_det, motion_val, last_box):
    """Runs detection (and species classification) on one sampled frame,
    annotates it, and fires an alert for any new, non-static detection."""
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
                            s_label = next((v for v in
                                (res.get('label'), res.get('prediction'),
                                 res.get('y_pred'))
                                if v is not None), "Unknown")
                            s_conf = next((v for v in
                                (res.get('confidence'), res.get('y_conf'))
                                if v is not None), 0.0)
                            if s_conf > SPECIES_THRESHOLD:
                                label = f"{obj_name}: {s_label} ({s_conf:.2f})"
                        except Exception as e:
                            logger.warning(f"[SYSTEM] Species "
                                           f"classification failed: {e}")
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
                        if (abs(x1 - lb[0]) < w * STATIC_TOLERANCE and
                            abs(y1 - lb[1]) < h * STATIC_TOLERANCE):
                            is_static = True
                    if not is_static:
                        with stats_lock:
                            stats[obj_name] = stats.get(obj_name, 0) + 1
                        last_box[d_key] = (x1, y1, x2, y2)
                        fname = f"{cam_id}_{obj_name}_{int(time.time())}.jpg"
                        fpath = os.path.join(BASE_OUTPUT_FOLDER,
                                             cam_id, fname)
                        cv2.imwrite(fpath, frame)
                        photo_executor.submit(send_telegram_photo, fpath,
                            f"ALERT: {label} on {cam_id}")
                        last_det[d_key] = time.time()
                    else:
                        logger.info(f"[FILTER] Static {obj_name} ignored "
                                   f"on {cam_id} at ({x1},{y1})")
    for c in [0, 1, 2]:
        motion_val[(cam_id, c)] = seen[c]

def ai_engine():
    """Processes frames: Detects objects and filters static false positives."""
    try:
        detector = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c",
                                               device="cpu", pretrained=True)
        # DFNE ("Deepfaune-New-England", a USGS-retrained variant of the
        # French Deepfaune model): unlike DeepfauneClassifier, its species
        # set is North American (bobcat, coyote, black bear, gray/red fox,
        # raccoon, skunk, white-tailed deer, wild turkey, etc.) -- matching
        # this project's actual use case.
        classifier = pw_classification.DFNE(device="cpu")
    except Exception as e:
        logger.critical("[SYSTEM] Failed to load AI models:", exc_info=True)
        send_telegram_message(f"Animal Catcher FAILED to start: "
                              f"could not load AI models ({e})")
        os._exit(1)
    last_det = {}
    motion_val = {}
    last_box = {}
    names = {0: "Animal", 1: "Person", 2: "Vehicle"}
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
    send_telegram_message(
        f"The animal catcher is online, version {VERSION}\n"
        f"Started: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        f"Streams: {len(CAMERA_CHANNELS)}")
    while True:
        cam_id, frame = detection_queue.get()
        try:
            _process_frame(cam_id, frame, detector, classifier, names,
                           colors, last_det, motion_val, last_box)
        except Exception as e:
            logger.error(f"[SYSTEM] ai_engine failed processing frame "
                        f"from {cam_id}: {e}")
            time.sleep(1)
        finally:
            detection_queue.task_done()

# --- 5. STARTUP ---
def _handle_shutdown(signum, frame):
    logger.info(f"[SYSTEM] Received signal {signum}, shutting down.")
    # Drop any not-yet-started uploads so shutdown isn't held up by the
    # whole queue; an upload already in flight can still delay exit by up
    # to its own request timeout.
    photo_executor.shutdown(wait=False, cancel_futures=True)
    sys.exit(0)

if __name__ == "__main__":
    print(f"Starting Animal Catcher v{VERSION}...")
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    for t in [ai_engine, summary_engine, cleanup_engine]:
        threading.Thread(target=t, name=t.__name__, daemon=True).start()
    for n in CAMERA_CHANNELS:
        threading.Thread(target=camera_thread, args=(n,), name=f"cam0{n}",
                         daemon=True).start()
        time.sleep(2)
    while True:
        time.sleep(1)

# EOF
