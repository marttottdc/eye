#!/usr/bin/env python3
"""
face_tracker_still.py  –  *alternate* test build

 • Uses a **dual-stream** approach with Picamera2:
     – 640×480 preview stream @30 fps for real-time face detection
     – Full-resolution still (sensor max) captured only when a face
       has waited SEND_EVERY seconds → crisp, no motion blur
 • Locks exposure to 1/200 s to kill smearing
 • Uploads **one** high-quality JPEG (Q=95) per burst via requests.post
 • Threaded worker + queue for non-blocking network I/O
 • Graceful shutdown on Ctrl-C or SIGTERM
"""

# ───────────────────────── imports ────────────────────────────────────────────
import base64, logging, os, queue, signal, threading, time
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Dict, Deque

import cv2, requests
from picamera2 import Picamera2

# ───────────────────────── config ─────────────────────────────────────────────
CASCADE_PATH   = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
DETECT_EVERY   = 5           # analyse 1 of N preview frames
SEND_EVERY     = 20          # seconds between uploads per track
MAX_MISS       = 15          # frames before dropping a track
MAX_QUEUE      = 8           # block producer if > N bursts pending
JPEG_Q         = 95          # quality for stills
HEARTBEAT_SEC  = 30

WEBHOOK_URL    = os.getenv("WEBHOOK_URL",   "<your-url>")
BEARER_TOKEN   = os.getenv("WEBHOOK_TOKEN", "<your-token>")

# ───────────────────────── logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tracker")

# ───────────────── dataclass & helpers ────────────────────────────────────────
@dataclass
class Track:
    bbox: Tuple[int, int, int, int]
    last_seen: float
    last_sent: float = 0.0
    miss: int = 0
    id: int = field(default_factory=int)

def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = aw * ah + bw * bh - inter
    return inter / union if union else 0.0

def encode_b64(img) -> str:
    buf = cv2.imencode(".jpg", img,
                       [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])[1]
    return base64.b64encode(buf).decode()

# ────────────── camera utilities (dual-stream) ───────────────────────────────
def init_camera() -> Tuple[Picamera2, dict, dict]:
    cam = Picamera2()

    preview_cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)},
        controls={"FrameRate": 30},
    )
    still_cfg = cam.create_still_configuration()  # max sensor resol.

    cam.configure(preview_cfg)
    cam.start()

    # one-shot AF / AE, then lock shutter to 1/200 s
    time.sleep(0.3)
    cam.set_controls({"AfMode": 2})          # run autofocus on HQ/v3
    cam.set_controls({"AeEnable": 1})
    time.sleep(0.5)                          # let AE settle
    cam.set_controls({"AeEnable": 0})
    cam.set_controls({"FrameDurationLimits": (5000, 5000)})  # 5 ms

    return cam, preview_cfg, still_cfg

def capture_still(cam: Picamera2, still_cfg: dict,
                  preview_cfg: dict) -> "np.ndarray":
    """Switch to still mode, grab image, switch back."""
    cam.switch_mode(still_cfg)
    time.sleep(0.1)                         # sensor settle
    img = cam.capture_array()
    cam.switch_mode(preview_cfg)
    return img

# ─────────────── worker + heartbeat threads ──────────────────────────────────
stop_ev = threading.Event()

def uploader(q: queue.Queue):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}",
               "Content-Type": "application/json"}
    while not stop_ev.is_set() or not q.empty():
        try:
            fid, b64_img = q.get(timeout=0.5)
        except queue.Empty:
            continue
        payload = {"id": fid, "img_b64": b64_img}
        delay = 2
        for attempt in range(5):
            try:
                r = requests.post(WEBHOOK_URL, headers=headers,
                                  json=payload, timeout=10)
                if r.ok:
                    log.info(f"[{fid}] sent still ({len(b64_img)//1024} kB)")
                    break
                raise RuntimeError(f"HTTP {r.status_code}")
            except Exception as e:
                log.warning(f"[{fid}] retry {attempt+1}: {e}")
                time.sleep(delay)
                delay *= 2
        else:
            log.error(f"[{fid}] dropped after 5 retries")
        q.task_done()

def heartbeat(q: queue.Queue, tracks: Dict[int, Track]):
    while not stop_ev.is_set():
        time.sleep(HEARTBEAT_SEC)
        log.debug(f"♥ heartbeat – queue={q.qsize()}  tracks={len(tracks)}")

# ─────────────────────────── main loop ───────────────────────────────────────
def track_and_send():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise FileNotFoundError(CASCADE_PATH)

    cam, prev_cfg, still_cfg = init_camera()
    frame_no = 0
    history: Deque = deque(maxlen=4)      # not used for still, but keeps flow
    tracks: Dict[int, Track] = {}
    next_id = 0

    q = queue.Queue(MAX_QUEUE)
    threading.Thread(target=uploader,  args=(q,), daemon=True).start()
    threading.Thread(target=heartbeat, args=(q, tracks), daemon=True).start()

    try:
        while not stop_ev.is_set():
            frame = cam.capture_array()          # preview 640×480
            frame_no += 1
            history.append(frame)

            if frame_no % DETECT_EVERY:
                continue

            small_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(small_gray, 1.2, 3, 0, (30, 30))

            # age out tracks
            for tr in tracks.values():
                tr.miss += 1

            # assign detections
            for (x, y, w, h) in faces:
                box = (x, y, w, h)
                best_iou, best_id = 0.0, None
                for tid, tr in tracks.items():
                    i = iou(box, tr.bbox)
                    if i > best_iou:
                        best_iou, best_id = i, tid
                if best_iou > 0.3:
                    tr = tracks[best_id]
                    tr.bbox, tr.miss, tr.last_seen = box, 0, time.time()
                    log.info(f"DETECT ▶ existing ID {best_id}")
                else:
                    tracks[next_id] = Track(box, time.time(), id=next_id)
                    log.info(f"DETECT ◎ new ID {next_id}")
                    next_id += 1

            # drop stale
            for tid in [tid for tid, tr in tracks.items() if tr.miss > MAX_MISS]:
                tracks.pop(tid, None)

            # maybe send still
            now = time.time()
            for tr in tracks.values():
                if now - tr.last_sent < SEND_EVERY or q.full():
                    continue
                hi = capture_still(cam, still_cfg, prev_cfg)
                q.put((tr.id, encode_b64(hi)))
                log.info(f"SEND ↥ queued ID {tr.id}")
                tr.last_sent = now

    finally:
        log.info("Shutting down…")
        stop_ev.set()
        q.join()
        cam.stop()

# ───────────────────────── entry-point ────────────────────────────────────────
def _signal_handler(sig, frame):
    log.info(f"Signal {sig} received – exiting…")
    stop_ev.set()

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

if __name__ == "__main__":
    track_and_send()
