#!/usr/bin/env python3

import configparser
import datetime
import glob
import logging
import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import cv2
import requests
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.models import detection as pw_detection

# --- 0. VERSIONING ---
VERSION = "0.12"

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
        cfg = {
            'user': parser.get('CAMERA', 'user'),
            'pass': parser.get('CAMERA', 'pass'),
            'ip': parser.get('CAMERA', 'ip'),
            'port': parser.get('CAMERA', 'port'),
            'telegram_token': parser.get('TELEGRAM', 'token'),
            'telegram_chat_id': parser.get('TELEGRAM', 'chat_id'),
            'base_output_folder': parser.get('PATHS', 'base_output_folder'),
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
            # Optional, defaulted rather than required: existing ac.cfg
            # files predate these keys, and a missing-key SystemExit would
            # otherwise break every deployment that hasn't added them yet.
            # speciesnet_model defaults to Google's own recommended model
            # identifier, which SpeciesNetClassifier downloads
            # automatically (via kagglehub, no account/credentials
            # needed for this public model -- confirmed by actually
            # downloading it) on first use, same as MegaDetector/DFNE's
            # own weights already do. Override with a local directory
            # path instead for a pre-downloaded, offline-friendly copy.
            'classifier': parser.get('DETECTION', 'classifier',
                                     fallback='dfne'),
            'speciesnet_model': parser.get(
                'DETECTION', 'speciesnet_model',
                fallback='kaggle:google/speciesnet/pyTorch/v4.0.3a/1'),
            # Extra margin added around MegaDetector's box before cropping
            # for species classification, as a fraction of the box's own
            # width/height. A tight box can clip a tail, ear, or antler
            # that the classifier needs -- doesn't affect the drawn
            # detection box or the static-position filter, only the
            # classifier's input crop.
            'crop_padding': parser.getfloat('DETECTION', 'crop_padding',
                                            fallback=0.15),
            'max_age_days': parser.getint('CLEANUP', 'max_age_days'),
            'cleanup_interval': parser.getint('CLEANUP', 'cleanup_interval'),
        }
        if cfg['classifier'] not in ('dfne', 'speciesnet'):
            raise ValueError(
                f"classifier must be 'dfne' or 'speciesnet', "
                f"got {cfg['classifier']!r}")
        return cfg
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
THRESHOLDS = _cfg['thresholds']
COOLDOWN = _cfg['cooldown']
FRAME_INTERVAL = _cfg['frame_interval']
SUMMARY_INTERVAL = _cfg['summary_interval']
SPECIES_THRESHOLD = _cfg['species_threshold']
# Tolerance for "Static" detection, as a fraction of frame width/height.
STATIC_TOLERANCE = _cfg['static_tolerance']
# Species classifier backend: "dfne" (default, PytorchWildlife's own,
# North American northeastern-US species set) or "speciesnet" (Google's
# 2498-taxa classifier, better suited to regions DFNE doesn't cover well
# -- e.g. no puma/mountain lion at all, and mule deer vs. DFNE's
# white-tailed deer only). See README's Species Classifier section.
CLASSIFIER_BACKEND = _cfg['classifier']
# Kaggle/HuggingFace model identifier or local directory -- passed
# straight through to SpeciesNetClassifier, which resolves either form
# itself. Unused when CLASSIFIER_BACKEND is "dfne".
SPECIESNET_MODEL = _cfg['speciesnet_model']
# Margin added around a detection box before cropping for species
# classification, as a fraction of the box's own width/height.
CROP_PADDING = _cfg['crop_padding']

# Cleanup Settings
MAX_AGE_DAYS = _cfg['max_age_days']
CLEANUP_INTERVAL = _cfg['cleanup_interval']

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ("rtsp_transport;tcp|"
                                               "stimeout;5000000")

# --- 2. LOGGING ---
# Writes to stdout rather than a separate log file: launchd already
# captures stdout to its own file (StandardOutPath in com.user.ac.plist),
# so a second, separately-rotated app log file was redundant. Rotation is
# now whatever's applied to that file outside this process (e.g. macOS
# newsyslog), not handled in-app.
logger = logging.getLogger("animalcatcher")
logger.setLevel(logging.INFO)
_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
logger.addHandler(_log_handler)
# PytorchWildlife pulls in ultralytics, which installs its own
# StreamHandler(stderr) on the ROOT logger at import time. Without this,
# every message logged here also propagates up and gets duplicated,
# unformatted, into stderr -- found by inspecting logging.getLogger()
# .handlers after `import ac`, not by guessing.
logger.propagate = False

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
    """Removes old snapshots every 24 hours."""
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
                report = (f"--- Animals Catcher Summary ---\n"
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

def _classify_dfne(classifier, crop):
    """Adapts PytorchWildlife's classifier API (DFNE, or any other
    pw_classification.* model) to a plain (label, confidence) pair."""
    s_res = classifier.single_image_classification(crop)
    res = s_res[0] if isinstance(s_res, list) else s_res
    label = next((v for v in
                  (res.get('label'), res.get('prediction'), res.get('y_pred'))
                  if v is not None), "Unknown")
    conf = next((v for v in (res.get('confidence'), res.get('y_conf'))
                 if v is not None), 0.0)
    return label, conf

def _classify_speciesnet(classifier, crop):
    """Adapts SpeciesNet's classifier API to a plain (label, confidence)
    pair. `crop` is a BGR numpy array (sliced directly out of a cv2
    frame); SpeciesNet expects RGB PIL images, so this converts first --
    getting that backwards would silently feed the model color-inverted
    crops instead of raising anything.

    Uses the bare classifier only (no detector/ensemble/geofencing from
    the full SpeciesNet pipeline): `crop` is already isolated to a
    MegaDetector animal detection, so no bounding box is passed to
    preprocess() -- it uses the whole (already-cropped) image as-is."""
    import PIL.Image
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    preprocessed = classifier.preprocess(PIL.Image.fromarray(rgb))
    result = classifier.predict(filepath="crop", img=preprocessed)
    classifications = result.get('classifications') or {}
    classes = classifications.get('classes')
    scores = classifications.get('scores')
    if not classes:
        return "Unknown", 0.0
    # Raw labels are "<uuid>;class;order;family;genus;species;common_name"
    # -- take the common name (last non-empty field); a higher-taxa or
    # non-species prediction (e.g. "mammalia" alone, "blank") leaves
    # fewer fields populated, so fall back to whatever's there.
    parts = [p for p in classes[0].split(';') if p]
    label = parts[-1].title() if parts else classes[0]
    return label, float(scores[0])

def _load_speciesnet_classifier():
    """Lazily imports speciesnet -- an optional dependency (see
    requirements-speciesnet.txt) not needed at all for the default DFNE
    backend, so a plain DFNE-only install never needs its much heavier
    dependency chain (pandas, matplotlib, kagglehub, ...).

    SPECIESNET_MODEL is passed straight through to SpeciesNetClassifier,
    which handles both forms itself: a kaggle: identifier (the default;
    auto-downloaded via kagglehub on first use, same as MegaDetector/
    DFNE's own weights -- confirmed this needs no Kaggle account for
    Google's public model) or a local directory already containing a
    pre-downloaded copy, for an offline-friendly deployment instead."""
    from speciesnet import SpeciesNetClassifier
    return SpeciesNetClassifier(SPECIESNET_MODEL, device="cpu")

def _process_frame(cam_id, frame, detector, classifier, names, colors,
                   last_det, motion_val, last_box, classify_fn=_classify_dfne):
    """Runs detection (and species classification) on one sampled frame,
    annotates it, and fires an alert for any new, non-static detection."""
    results = detector.single_image_detection(frame)
    det = results.get("detections")
    seen = {0: False, 1: False, 2: False}
    if det is not None and len(det.confidence) > 0:
        h, w, _ = frame.shape
        for i in range(len(det.confidence)):
            conf, cls = float(det.confidence[i]), int(det.class_id[i])
            obj_name = names.get(cls, "Object")
            logger.info(f"[DETECT] {obj_name} conf={conf:.2f} on {cam_id}")
            if conf > THRESHOLDS.get(cls, 0.5):
                seen[cls] = True
                label = f"{obj_name} ({conf:.2f})"
                box = det.xyxy[i]
                x1, y1, x2, y2 = (int(box[0]), int(box[1]),
                                  int(box[2]), int(box[3]))
                if cls == 0 and conf > SPECIES_THRESHOLD:
                    # Pad the box before cropping for classification --
                    # a tight box can clip a tail, ear, or antler the
                    # classifier needs. Only affects this crop; x1/y1/x2/y2
                    # themselves (the drawn box, the static-position
                    # filter) are untouched.
                    pad_x = int((x2 - x1) * CROP_PADDING)
                    pad_y = int((y2 - y1) * CROP_PADDING)
                    crop = frame[max(0, y1 - pad_y):min(h, y2 + pad_y),
                                 max(0, x1 - pad_x):min(w, x2 + pad_x)]
                    if crop.size > 0:
                        try:
                            s_label, s_conf = classify_fn(classifier, crop)
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

def _clean_stale_model_downloads():
    """MegaDetectorV6's weights are fetched via the `wget` package, which
    downloads into a `<prefix>.tmp` file in the current working directory
    and only renames it to the final filename on success. A crash or
    restart mid-download (e.g. during an earlier AI-model-load failure)
    leaves that file behind forever, since nothing else ever cleans it up."""
    for path in glob.glob("MDV6*.tmp"):
        try:
            os.remove(path)
            logger.info(f"[SYSTEM] Removed stale partial download: {path}")
        except OSError as e:
            logger.warning(f"[SYSTEM] Could not remove stale partial "
                           f"download {path}: {e}")

def ai_engine():
    """Processes frames: Detects objects and filters static false positives."""
    _clean_stale_model_downloads()
    try:
        detector = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c",
                                               device="cpu", pretrained=True)
        if CLASSIFIER_BACKEND == 'speciesnet':
            classifier = _load_speciesnet_classifier()
            classify_fn = _classify_speciesnet
        else:
            # DFNE ("Deepfaune-New-England", a USGS-retrained variant of
            # the French Deepfaune model): unlike DeepfauneClassifier,
            # its species set is North American (bobcat, coyote, black
            # bear, gray/red fox, raccoon, skunk, white-tailed deer,
            # wild turkey, etc.) -- matching this project's original
            # (northeastern US) use case. See README's Species
            # Classifier section for when speciesnet fits better.
            classifier = pw_classification.DFNE(device="cpu")
            classify_fn = _classify_dfne
    except Exception as e:
        logger.critical("[SYSTEM] Failed to load AI models:", exc_info=True)
        send_telegram_message(f"Animals Catcher FAILED to start: "
                              f"could not load AI models ({e})")
        os._exit(1)
    last_det = {}
    motion_val = {}
    last_box = {}
    names = {0: "Animal", 1: "Person", 2: "Vehicle"}
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
    send_telegram_message(
        f"The Animals Catcher is online, version {VERSION}\n"
        f"Started: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        f"Streams: {len(CAMERA_CHANNELS)}")
    while True:
        cam_id, frame = detection_queue.get()
        try:
            _process_frame(cam_id, frame, detector, classifier, names,
                           colors, last_det, motion_val, last_box,
                           classify_fn)
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

def _handle_log_reopen(signum, frame):
    """SIGUSR1: reopen stdout/stderr in place, for use after an external
    tool (e.g. newsyslog) rotates the files launchd's StandardOutPath/
    StandardErrorPath redirect to. Without this, this process's inherited
    fds would keep appending to the now-renamed, archived file forever --
    neither launchd nor this process otherwise notices the rename, and
    nothing reopens StandardOutPath short of a full restart. Requires
    AC_STDOUT_LOG/AC_STDERR_LOG (set in com.user.ac.plist) to know which
    paths to reopen; a plain `python3 ac.py` run without them is a no-op
    here."""
    sys.stdout.flush()
    sys.stderr.flush()
    for env_var, fd in (('AC_STDOUT_LOG', 1), ('AC_STDERR_LOG', 2)):
        path = os.environ.get(env_var)
        if not path:
            continue
        try:
            new_fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            os.dup2(new_fd, fd)
            os.close(new_fd)
        except OSError as e:
            logger.warning(f"[SYSTEM] Failed to reopen fd {fd} ({path}) "
                           f"after log rotation: {e}")
    logger.info(f"[SYSTEM] Reopened stdout/stderr after external log "
               f"rotation (signal {signum}).")

def _write_pid_file():
    """Lets an external tool (e.g. newsyslog, via its pid_file/
    signal_number config fields) find this process to signal after
    rotating its log files. Best-effort: a daemon that can't write this
    still runs fine, it just can't be told to reopen its logs without a
    full restart."""
    try:
        with open(os.path.join(os.getcwd(), 'ac.pid'), 'w') as f:
            f.write(str(os.getpid()))
    except OSError as e:
        logger.warning(f"[SYSTEM] Could not write PID file: {e}")

if __name__ == "__main__":
    print(f"Starting Animals Catcher v{VERSION}...")
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGUSR1, _handle_log_reopen)
    _write_pid_file()
    for t in [ai_engine, summary_engine, cleanup_engine]:
        threading.Thread(target=t, name=t.__name__, daemon=True).start()
    for n in CAMERA_CHANNELS:
        threading.Thread(target=camera_thread, args=(n,), name=f"cam0{n}",
                         daemon=True).start()
        time.sleep(2)
    while True:
        time.sleep(1)

# EOF
