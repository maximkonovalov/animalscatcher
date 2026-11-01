#!/usr/bin/env python3

import collections
import configparser
import datetime
import glob
import logging
import os
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
VERSION = "0.20"

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
            # Motion check tunings, the same for every camera -- see
            # MotionDetector and README's Motion Detection section.
            'motion_threshold': parser.getfloat('DETECTION',
                                                'motion_threshold',
                                                fallback=0.0012),
            'motion_pixel_delta': parser.getint('DETECTION',
                                                'motion_pixel_delta',
                                                fallback=20),
            'motion_check_interval': parser.getint('DETECTION',
                                                   'motion_check_interval',
                                                   fallback=5),
            'motion_lookback': parser.getfloat('DETECTION', 'motion_lookback',
                                               fallback=2.0),
            # How long (seconds) a sighting still counts as "the previous
            # frame" for the alert gate's two-frame latch. Frames used to
            # arrive around the clock; now they only follow motion, so the
            # previous look at a camera can be hours old and must not
            # confirm a new sighting -- see _expire_stale_latch.
            'confirm_window': parser.getfloat('DETECTION', 'confirm_window',
                                              fallback=30.0),
            # frame_interval used to be required: every Nth raw frame went
            # to the AI whether or not anything moved. That sampling is
            # gone (frames are analysed when motion is detected), but
            # existing ac.cfg files still carry the key, so it is
            # tolerated -- __main__ warns that it no longer does anything.
            'frame_interval_set': parser.has_option('DETECTION',
                                                    'frame_interval'),
            'max_age_days': parser.getint('CLEANUP', 'max_age_days'),
            'cleanup_interval': parser.getint('CLEANUP', 'cleanup_interval'),
        }
        # Optional, defaulted: the defaults are the layout this project
        # was originally hardcoded for (a Hikvision-style DVR with cameras
        # on channels 4-6, each read from its sub-stream), so an existing
        # ac.cfg keeps working unchanged. rtsp_path is everything after
        # host:port; str.format fills {channel}, so {channel:02d} works
        # for recorders whose paths expect zero-padded numbers.
        raw_channels = parser.get('CAMERA', 'channels', fallback='4,5,6')
        rtsp_path = parser.get('CAMERA', 'rtsp_path',
                               fallback='/Streaming/Channels/{channel}02')
        try:
            channels = [int(c) for c in raw_channels.split(',') if c.strip()]
        except ValueError:
            raise ValueError("channels must be comma-separated integers, "
                             f"got {raw_channels!r}") from None
        if not channels:
            raise ValueError("channels must list at least one channel number")
        if len(set(channels)) != len(channels):
            raise ValueError(f"channels lists a number twice: {channels}")
        if not rtsp_path.startswith('/'):
            raise ValueError(f"rtsp_path must start with '/', "
                             f"got {rtsp_path!r}")
        try:
            urls = {rtsp_path.format(channel=c) for c in channels}
        except Exception as e:
            # str.format raises a different type for each kind of broken
            # template ({typo}: KeyError, {}: IndexError, {channel:x}:
            # ValueError, ...); here they all just mean a bad rtsp_path.
            raise ValueError(f"rtsp_path {rtsp_path!r} is not a valid "
                             f"template, {{channel}} is the only "
                             f"placeholder allowed: {e!r}") from None
        if len(urls) != len(channels):
            # Every channel would open the same stream: N threads
            # decoding one camera and alerting N times per detection.
            raise ValueError(f"rtsp_path {rtsp_path!r} must contain "
                             "{channel} when more than one channel is "
                             "configured")
        cfg['channels'] = channels
        cfg['rtsp_path'] = rtsp_path
        # A value outside its range fails here rather than later inside a
        # camera thread -- a motion_check_interval of 0, say, would be a
        # ZeroDivisionError on every frame.
        for key, in_range, wanted in (
                ('motion_threshold', lambda v: 0 <= v < 1,
                 "at least 0 and below 1 (a fraction of the frame)"),
                ('motion_pixel_delta', lambda v: 1 <= v <= 254,
                 "between 1 and 254"),
                ('motion_check_interval', lambda v: v >= 1, "at least 1"),
                ('motion_lookback', lambda v: 0 <= v <= 60,
                 "between 0 and 60 seconds"),
                ('confirm_window', lambda v: v > 0, "above 0 seconds")):
            if not in_range(cfg[key]):
                raise ValueError(f"{key} must be {wanted}, got {cfg[key]}")
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
CAMERA_CHANNELS = _cfg['channels']
RTSP_PATH = _cfg['rtsp_path']
TELEGRAM_TOKEN = _cfg['telegram_token']
TELEGRAM_CHAT_ID = _cfg['telegram_chat_id']
# A named constant rather than a literal embedded in each f-string below:
# CodeQL's incomplete-url-substring-sanitization check flags an f-string
# with a domain literal sitting next to an interpolated variable, even
# though TELEGRAM_TOKEN only ever lands in the path (after the domain),
# never able to affect the host -- this doesn't change that (it was
# already safe), just separates the static and dynamic parts more
# clearly for both readers and static analysis.
TELEGRAM_API_BASE = "https://api.telegram.org"

# Path & Detection Settings
BASE_OUTPUT_FOLDER = _cfg['base_output_folder']
THRESHOLDS = _cfg['thresholds']
COOLDOWN = _cfg['cooldown']
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
# Motion check tunings (see MotionDetector and camera_thread): how much of
# the frame must change, how much a pixel must change, how often to check,
# and how far back the reference frame is.
MOTION_THRESHOLD = _cfg['motion_threshold']
MOTION_PIXEL_DELTA = _cfg['motion_pixel_delta']
MOTION_CHECK_INTERVAL = _cfg['motion_check_interval']
MOTION_LOOKBACK = _cfg['motion_lookback']
# Side of the blur kernel applied to both frames before they are compared.
# Not a setting: 5 beat 3 on every measure when the defaults were calibrated
# (same alerts, a fraction of the dusk noise).
MOTION_BLUR = 5
# Seconds a sighting keeps counting as "the previous frame" for the alert
# gate's two-frame latch -- see _expire_stale_latch.
CONFIRM_WINDOW = _cfg['confirm_window']

# Cleanup Settings
MAX_AGE_DAYS = _cfg['max_age_days']
CLEANUP_INTERVAL = _cfg['cleanup_interval']

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ("rtsp_transport;tcp|"
                                               "stimeout;5000000")

# --- 2. LOGGING ---
# Writes to stdout rather than a separate log file: launchd already
# captures stdout to its own file (StandardOutPath in assets/com.user.ac.plist),
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
# How many frames wait for ai_engine, all cameras together. While one frame
# is analysed (about 3.5 s on the target hardware) a camera that keeps showing
# motion flags four more every second, so this is the room a spike of motion
# gets before frames are dropped -- and the summary's drop count says whether
# it was enough.
AI_QUEUE_SIZE = 15

class FrameQueue:
    """The hand-off from the camera threads to ai_engine: one queue of the
    frames that showed motion, shared by every camera, oldest first.
    ai_engine is a single thread, so a spike of motion queues frames faster
    than it takes them; the queue is the room that spike gets. When it is
    full the oldest frame is dropped to make room for the new one, and
    offer() says which camera's frame that was so the caller can count it."""

    def __init__(self, size):
        self._size = size
        self._cond = threading.Condition()
        self._frames = collections.deque()  # (cam_id, frame, time offered)

    def offer(self, cam_id, frame):
        """Queues `frame` behind the ones already waiting. If the queue was
        full, returns the camera id of the oldest frame, dropped to make
        room; otherwise None."""
        with self._cond:
            dropped = None
            if len(self._frames) >= self._size:
                dropped = self._frames.popleft()[0]
            self._frames.append((cam_id, frame, time.time()))
            self._cond.notify()
        return dropped

    def take(self):
        """Blocks until a frame is waiting, then returns the oldest as
        (cam_id, frame, time it was offered)."""
        with self._cond:
            while not self._frames:
                self._cond.wait()
            return self._frames.popleft()

detection_queue = FrameQueue(AI_QUEUE_SIZE)

stats_lock = threading.Lock()
stats = {
    "Animal": 0, "Person": 0, "Vehicle": 0,
    # Per-camera counters behind the summary's MOTION block, by cam_id;
    # see _motion_stats_for.
    "motion": {},
    "start_time": datetime.datetime.now(),
    "streams": {}
}

def _new_motion_stats():
    return {"checks": 0, "motion": 0, "analysed": 0, "animals": 0,
            "dropped": 0, "peak": 0.0, "wait_sum": 0.0, "wait_max": 0.0}

def _motion_stats_for(cam_id):
    """A camera's counters since the last summary. Call with stats_lock
    held."""
    return stats["motion"].setdefault(cam_id, _new_motion_stats())

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
    url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        logger.warning(f"[TELEGRAM] Failed to send message: {_redact(str(e))}")

def send_telegram_photo(photo_path, caption):
    """Best-effort notification: never raises, same contract as
    send_telegram_message."""
    url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_TOKEN}/sendPhoto"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    try:
        with open(photo_path, "rb") as photo:
            requests.post(url, data=payload, files={"photo": photo}, timeout=15)
    except Exception as e:
        logger.warning(f"[TELEGRAM] Failed to send photo {photo_path}: "
                       f"{_redact(str(e))}")

# --- 4. ENGINE THREADS ---

def cleanup_engine():
    """Removes snapshots older than max_age_days: once at startup, then
    every cleanup_interval hours."""
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

def _motion_report(motion_stats):
    """The summary's per-camera MOTION lines, from the counters in
    `motion_stats` (a camera that counted nothing yet reads as zeros).
    The strongest change is shown beside the threshold because that is
    what tuning is about: a peak far below the threshold means the camera
    is never triggering, a peak just above it that the threshold sits at
    the noise."""
    lines = []
    for channel in CAMERA_CHANNELS:
        cam_id = _cam_id(channel)
        m = motion_stats.get(cam_id) or _new_motion_stats()
        line = (f"- {cam_id}: {m['checks']} checks, {m['motion']} motion, "
                f"{m['analysed']} analysed ({m['animals']} with an animal), "
                f"{m['dropped']} dropped (queue full)\n"
                f"  strongest change {m['peak']:.3%} (threshold "
                f"{MOTION_THRESHOLD:.3%})")
        if m['analysed']:
            line += (f"; AI wait avg {m['wait_sum'] / m['analysed']:.1f}s, "
                     f"max {m['wait_max']:.1f}s")
        lines.append(line)
    return "\n".join(lines)

def _queue_report(motion_stats):
    """The summary's AI QUEUE lines: how many frames that showed motion
    were offered to the queue, over every camera, and how many were dropped
    from it to make room. Dropped frames are the oldest waiting, mostly
    near-copies from the early part of a long burst, so the share says how
    much more motion there was than the AI could take, not how many events
    it missed."""
    offered = sum(m["motion"] for m in motion_stats.values())
    dropped = sum(m["dropped"] for m in motion_stats.values())
    share = dropped / offered * 100 if offered else 0.0
    return (f"- Offered: {offered} (frames that showed motion)\n"
            f"- Dropped (queue full): {dropped} ({share:.1f}%)")

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
                          f"- Vehicles: {stats['Vehicle']}\n\n"
                          f"MOTION:\n{_motion_report(stats['motion'])}\n\n"
                          f"AI QUEUE (room for {AI_QUEUE_SIZE} frames, all "
                          f"cameras):\n{_queue_report(stats['motion'])}")
                stats.update({"Animal": 0, "Person": 0, "Vehicle": 0,
                              "motion": {}, "start_time": now})
            send_telegram_message(report)
        except Exception as e:
            logger.error(f"[SYSTEM] summary_engine iteration failed: {e}")

# The size every frame is shrunk to before two frames are compared, as
# (width, height). Small enough that a check costs a millisecond or two on a
# 720p frame; a small or distant animal is only a few pixels of it, which is
# why the thresholds that detect one are fractions of a percent.
MOTION_SIZE = (160, 120)

class MotionDetector:
    """Frame-differencing motion check for ONE camera -- far cheaper than
    a MegaDetector pass, which is why it decides what gets one. Each
    frame is shrunk, turned gray and blurred (to suppress sensor and
    compression noise), then compared with a reference frame from
    `lookback` seconds earlier (0 = the previous check). A pixel counts as
    changed when it differs by more than `pixel_delta` levels, and the
    frame counts as showing motion when the largest connected patch of
    changed pixels covers more than `threshold` of it. Taking the largest
    patch rather than the total changed area is what keeps scattered
    noise from adding up to motion, so a small threshold stays usable.

    A longer lookback makes slow movers visible: a deer that shifts a
    pixel between two frames a quarter second apart has moved several by
    the time of a frame two seconds older. The price is that a mover
    keeps registering for about `lookback` seconds after it stops, or
    after it has left.
    """

    def __init__(self, threshold, pixel_delta, lookback, blur=MOTION_BLUR):
        self.threshold = threshold
        self.pixel_delta = pixel_delta
        self.lookback = lookback
        self.blur = blur
        self._history = collections.deque()   # (time, gray), oldest first

    def reset(self):
        """Forget every earlier frame -- after a reconnect, so the first
        frame back isn't compared with one from before the outage."""
        self._history.clear()

    def check(self, frame, now=None):
        """Compares `frame` (a BGR array, the newer of the two frames
        compared) with the reference and returns (moved, size): whether it
        shows motion, and the largest changed patch as a fraction of the
        frame. The first frame after construction or reset has nothing to
        be compared with and never counts as motion."""
        if now is None:
            now = time.time()
        # Plain bilinear shrinking. INTER_AREA averages more pixels per
        # output pixel, but costs about eight times as much on a 720p
        # frame (measured: 9.8 ms against 1.2), which matters on the
        # hardware this runs on, and its noise floor on a quiet night
        # scene was no lower.
        small = cv2.resize(frame, MOTION_SIZE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.blur > 1:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        history = self._history
        # The reference is the newest frame at least `lookback` old, or,
        # until the history is that long, the oldest there is.
        while len(history) > 1 and now - history[1][0] >= self.lookback:
            history.popleft()
        reference = history[0][1] if history else None
        history.append((now, gray))
        if reference is None:
            return False, 0.0
        diff = cv2.absdiff(gray, reference)
        _, mask = cv2.threshold(diff, self.pixel_delta, 255,
                                cv2.THRESH_BINARY)
        count, _, areas, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        size = (areas[1:, cv2.CC_STAT_AREA].max() / mask.size
                if count > 1 else 0.0)
        return bool(size > self.threshold), float(size)

def _cam_id(channel):
    """Zero-padded so the default channels (4, 5, 6) keep their original
    cam04/cam05/cam06 snapshot folders, and 10+ stay tidy (cam10, not
    cam010)."""
    return f"cam{channel:02d}"

def _rtsp_url(channel):
    """RTSP URL for one channel. Credentials are percent-encoded, since a
    raw @, :, / or # in a username/password would be parsed as URL
    structure. The result embeds those credentials, so it must never be
    logged -- hand it straight to cv2.VideoCapture."""
    user = quote(USER, safe='')
    password = quote(PASS, safe='')
    return (f"rtsp://{user}:{password}@{IP}:{PORT}"
            f"{RTSP_PATH.format(channel=channel)}")

# After this many seconds without motion on a camera, its next motion is
# logged as a new event; within a burst only the first trigger is logged.
MOTION_LOG_GAP = 10

def camera_thread(cam_num):
    """Reads one camera's RTSP stream and watches it for motion with a
    detector of its own. When a check finds motion, the frame it just looked
    at -- the later of the two it compared -- is put in detection_queue, the
    one queue every camera shares with ai_engine; a frame that shows no
    motion never reaches the AI. If the queue was full its oldest frame was
    dropped to make room, and that is counted against the camera it came
    from."""
    cam_id = _cam_id(cam_num)
    url = _rtsp_url(cam_num)
    detector = MotionDetector(MOTION_THRESHOLD, MOTION_PIXEL_DELTA,
                              MOTION_LOOKBACK, MOTION_BLUR)
    os.makedirs(os.path.join(BASE_OUTPUT_FOLDER, cam_id), exist_ok=True)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    f_idx = 0
    last_motion = 0.0
    while True:
        try:
            f_idx += 1
            check = f_idx % MOTION_CHECK_INTERVAL == 0
            # Only every motion_check_interval-th raw frame is checked --
            # cheap per call, but continuously across every camera thread
            # it measurably competes with ai_engine for CPU on constrained
            # hardware. grab() decodes a frame without converting it to a
            # numpy array, so the frames that aren't checked skip that
            # conversion too.
            ok = cap.grab()
            frame = None
            if ok and check:
                ok, frame = cap.retrieve()
                ok = ok and frame is not None
            if not ok:
                with stats_lock:
                    stats["streams"][cam_id] = {"status": "OFFLINE",
                                                "res": "N/A"}
                cap.release()
                detector.reset()
                time.sleep(5)
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                continue
            if not check:
                continue
            h, w = frame.shape[:2]
            now = time.time()
            moved, size = detector.check(frame, now)
            with stats_lock:
                stats["streams"][cam_id] = {"status": "ONLINE",
                                            "res": f"{w}x{h}"}
                counts = _motion_stats_for(cam_id)
                counts["checks"] += 1
                counts["peak"] = max(counts["peak"], size)
                if moved:
                    counts["motion"] += 1
            if moved:
                if now - last_motion > MOTION_LOG_GAP:
                    logger.info(f"[MOTION] {cam_id}: {size:.3%} of the frame "
                                f"changed (threshold {MOTION_THRESHOLD:.3%})")
                last_motion = now
                dropped = detection_queue.offer(cam_id, frame)
                if dropped:
                    with stats_lock:
                        _motion_stats_for(dropped)["dropped"] += 1
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
    """Runs detection (and species classification) on one frame that showed
    motion, annotates it, counts it for the summary if an animal is on it,
    and fires an alert for any new, non-static detection."""
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
    if seen[0]:
        # One more motion frame with an animal on it, whatever the alert
        # gate made of it -- what the summary's MOTION block reports as how
        # often the motion check pays off.
        with stats_lock:
            _motion_stats_for(cam_id)["animals"] += 1

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

def _expire_stale_latch(motion_val, last_look, cam_id, seen_at):
    """Forgets a camera's two-frame latch when its previous look is more
    than CONFIRM_WINDOW seconds older than the one at `seen_at`, then
    records `seen_at` as its latest. The latch means "also seen on the
    previous processed frame"; frames used to arrive around the clock, but
    they now only follow motion, so the previous frame can be hours old --
    and a sighting then must not count as already confirmed by one from
    the last time something walked past."""
    if seen_at - last_look.get(cam_id, seen_at) > CONFIRM_WINDOW:
        for cls in (0, 1, 2):
            motion_val[(cam_id, cls)] = False
    last_look[cam_id] = seen_at

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
    last_look = {}
    names = {0: "Animal", 1: "Person", 2: "Vehicle"}
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
    send_telegram_message(
        f"The Animals Catcher is online, version {VERSION}\n"
        f"Started: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        f"Streams: {len(CAMERA_CHANNELS)}")
    while True:
        cam_id, frame, offered_at = detection_queue.take()
        waited = time.time() - offered_at
        with stats_lock:
            counts = _motion_stats_for(cam_id)
            counts["analysed"] += 1
            counts["wait_sum"] += waited
            counts["wait_max"] = max(counts["wait_max"], waited)
        try:
            _expire_stale_latch(motion_val, last_look, cam_id, offered_at)
            _process_frame(cam_id, frame, detector, classifier, names,
                           colors, last_det, motion_val, last_box,
                           classify_fn)
        except Exception as e:
            logger.error(f"[SYSTEM] ai_engine failed processing frame "
                        f"from {cam_id}: {e}")
            time.sleep(1)

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
    AC_STDOUT_LOG/AC_STDERR_LOG (set in assets/com.user.ac.plist) to know which
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
    if _cfg['frame_interval_set']:
        logger.warning("[SYSTEM] frame_interval in ac.cfg no longer does "
                       "anything: frames reach the AI only when motion is "
                       "detected (see README, Motion Detection). Remove it.")
    logger.info(f"[SYSTEM] Motion check: threshold {MOTION_THRESHOLD:.3%}, "
                f"pixel delta {MOTION_PIXEL_DELTA}, every "
                f"{MOTION_CHECK_INTERVAL} frames, lookback "
                f"{MOTION_LOOKBACK:g}s")
    for t in [ai_engine, summary_engine, cleanup_engine]:
        threading.Thread(target=t, name=t.__name__, daemon=True).start()
    for n in CAMERA_CHANNELS:
        threading.Thread(target=camera_thread, args=(n,), name=_cam_id(n),
                         daemon=True).start()
        time.sleep(2)
    while True:
        time.sleep(1)

# EOF
