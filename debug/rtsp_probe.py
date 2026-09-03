#!/usr/bin/env python3
"""Standalone RTSP + cv2.VideoCapture connectivity probe, independent of
ac.py and PytorchWildlife (no AI models loaded, so this isolates network
behavior from the earlier AI-model-loading issues entirely).

Built to compare identities for the still-unresolved UserName privilege
drop investigation (see RELEASES.md v0.8's UNRESOLVED note): run this
same interpreter (python3.10) once as root and once as a dropped-
privilege user via the two throwaway plists in this directory, then
diff their output.

Reads camera credentials from the same ac.cfg this daemon already uses
(via AC_CONFIG_PATH, or an explicit path as argv[1]) rather than taking
them as a CLI argument or plist EnvironmentVariables entry, so nothing
here ever puts real credentials in argv (visible to any local user via
`ps`) or in a file that could end up committed to git.
"""
import configparser
import os
import pwd
import socket
import sys
import time
from urllib.parse import quote

import cv2


def load_camera_config(path):
    parser = configparser.ConfigParser()
    if not parser.read(path):
        sys.exit(f"Config not found or unreadable: {path}")
    return (parser.get('CAMERA', 'ip'), parser.get('CAMERA', 'port'),
            parser.get('CAMERA', 'user'), parser.get('CAMERA', 'pass'))


def report_identity():
    print(f"[IDENTITY] uid={os.getuid()} euid={os.geteuid()} "
         f"user={pwd.getpwuid(os.getuid()).pw_name} "
         f"cwd={os.getcwd()} home={os.environ.get('HOME')}", flush=True)


def probe_tcp(ip, port, timeout=5):
    start = time.time()
    try:
        with socket.create_connection((ip, int(port)), timeout=timeout):
            print(f"[TCP] connect to {ip}:{port} -> OK in "
                 f"{time.time() - start:.2f}s", flush=True)
    except OSError as e:
        print(f"[TCP] connect to {ip}:{port} -> FAILED in "
             f"{time.time() - start:.2f}s ({e})", flush=True)


def probe_rtsp(ip, port, user, password, timeout_us=8_000_000):
    # Same FFmpeg options ac.py sets, so this reproduces its actual
    # capture behavior rather than cv2's untuned defaults.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;tcp|stimeout;{timeout_us}")
    url = f"rtsp://{quote(user)}:{quote(password)}@{ip}:{port}"
    start = time.time()
    cap = cv2.VideoCapture(url)
    opened = cap.isOpened()
    frame_ok = False
    if opened:
        frame_ok = cap.read()[0]
    cap.release()
    print(f"[RTSP] cv2.VideoCapture -> opened={opened} "
         f"frame_read={frame_ok} in {time.time() - start:.2f}s", flush=True)


def main():
    config_path = (sys.argv[1] if len(sys.argv) > 1
                   else os.environ.get('AC_CONFIG_PATH',
                                       '/Users/maxim/nvr/ac.cfg'))
    report_identity()
    ip, port, user, password = load_camera_config(config_path)
    probe_tcp(ip, port)
    probe_rtsp(ip, port, user, password)


if __name__ == "__main__":
    main()
