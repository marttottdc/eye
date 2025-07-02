#!/usr/bin/env python3
"""
Face-tracker / uploader for Raspberry Pi 3 B+.
— Haar detection every N frames on a down-scaled copy
— IoU tracking for stable IDs
— Five JPEGs per ID (n-2…n+2) POSTed via requests
— Console logging + heartbeat
"""

import base64, logging, os, queue, signal, threading, time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2, requests
from picamera2 import Picamera2

# ───────────────────────────── config ─────────────────────────────────────────
CASCADE_PATH   = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
DETECT_EVERY   = 5            # analyse 1 out of N frames
SEND_EVERY     = 20           # seconds between uploads per face ID
MAX_MISS       = 15           # frames before a track is dropped
MAX_QUEUE      = 8            # back-pressure – producer blocks when > N batches
JPEG_Q         = 72           # JPEG quality (0-100)
HEARTBEAT_SEC  = 30           # heartbeat interval

WEBHOOK_URL    = "https://developer.moio.ai/webhooks/f8284f2f-db74-4b45-a1e1-30479ac3117c/"
BEARER_TOKEN   = "e64465fbe22356fd66b429749a85c4783eb1262f5c763145ba4cb24585eb84f6"

# ────────────────────────── logging setup ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tracker")

# ─────────────────────────── dataclasses ──────────────────────────────────────
@dataclass
class Track:
    bbox: Tuple[int, int, int, int]
    last_seen: float
    last_sent: float = 0.0
    miss: int = 0
    id: int = field(default_factory=int)

# ───────────────────────── helper functions ───────────────────────────────────


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union of two (x,y,w,h) rectangles."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def create_camera() -> Picamera2:
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}))
    cam.start()
    return cam


def encode_images(imgs) -> List[str]:
    """JPEG-encode & b64 each image."""
    return [
        base64.b64encode(
            cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])[1]
        ).decode()
        for im in imgs
    ]

# ─────────────────────── worker & heartbeat threads ───────────────────────────


def uploader(q: queue.Queue, stop: threading.Event):
    """Blocking worker draining q and POSTing bursts via requests."""
    sess = requests.Session()
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}",
               "Content-Type": "application/json"}
    while not stop.is_set() or not q.empty():
        try:
            fid, imgs = q.get(timeout=0.5)
        except queue.Empty:
            continue

        b64_array = encode_images(imgs)
        tries, delay = 0, 2
        while tries < 5:
            try:
                r = sess.post(
                    WEBHOOK_URL,
                    headers=headers,
                    json={"id": fid, "img_b64_array": b64_array},
                    timeout=10,
                )
                if r.ok:
                    log.info(f"[{fid}] sent {len(imgs)} imgs")
                    break
                raise RuntimeError(f"HTTP {r.status_code}")
            except Exception as e:
                tries += 1
                log.warning(f"[{fid}] retry {tries}/5: {e}")
                time.sleep(delay)
                delay *= 2
        else:
            log.error(f"[{fid}] dropped after 5 retries")
        q.task_done()


def heartbeat(q: queue.Queue, tracks: Dict[int, Track], stop: threading.Event):
    while not stop.is_set():
        time.sleep(HEARTBEAT_SEC)
        log.debug(f"♥ heartbeat – queue={q.qsize()} tracks={len(tracks)}")

# ──────────────────────────── main loop ───────────────────────────────────────


def track_and_send():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise FileNotFoundError(CASCADE_PATH)

    cam             = create_camera()
    frame_no        = 0
    last4           = deque(maxlen=4)          # previous 4 frames
    tracks: Dict[int, Track] = {}
    next_id         = 0

    q       = queue.Queue(MAX_QUEUE)
    stop_ev = threading.Event()

    worker  = threading.Thread(target=uploader,  args=(q, stop_ev), daemon=True)
    hb      = threading.Thread(target=heartbeat, args=(q, tracks, stop_ev), daemon=True)
    worker.start(), hb.start()

    try:
        while True:
            high_res = cam.capture_array()
            frame_no += 1
            last4.append(high_res)

            if frame_no % DETECT_EVERY:
                continue

            # detection (down-scale for speed)
            small  = cv2.resize(high_res, (320, 240))
            gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces  = cascade.detectMultiScale(gray, 1.2, 3, 0, (30, 30))
            sx, sy = high_res.shape[1] / 320, high_res.shape[0] / 240
            faces  = [(int(x * sx), int(y * sy), int(w * sx), int(h * sy))
                      for x, y, w, h in faces]

            # age tracks
            for tr in tracks.values():
                tr.miss += 1

            # assign detections
            for box in faces:
                best_iou, best_id = 0.0, None
                for tid, tr in tracks.items():
                    val = iou(box, tr.bbox)
                    if val > best_iou:
                        best_iou, best_id = val, tid
                if best_iou > 0.3:
                    tr = tracks[best_id]
                    tr.bbox, tr.miss, tr.last_seen = box, 0, time.time()
                    log.info(f"DETECT ▶︎ existing ID {best_id}")
                else:
                    tracks[next_id] = Track(box, time.time(), id=next_id)
                    log.info(f"DETECT ◎ new ID {next_id}")
                    next_id += 1

            # drop stale
            for tid in [tid for tid, tr in tracks.items() if tr.miss > MAX_MISS]:
                tracks.pop(tid, None)

            # enqueue bursts
            now = time.time()
            for tr in tracks.values():
                if now - tr.last_sent < SEND_EVERY:
                    continue
                if q.full():
                    continue
                burst = list(last4) + [cam.capture_array(), cam.capture_array()]
                q.put((tr.id, burst))
                log.info(f"SEND  ↥ queued ID {tr.id} ({len(burst)} imgs)")
                tr.last_sent = now
    except KeyboardInterrupt:
        log.info("Ctrl-C – shutting down…")
    finally:
        stop_ev.set()
        q.join()           # wait until everything flushed
        cam.stop()
        worker.join()
        hb.join()

# ───────────────────────── entry-point ────────────────────────────────────────


def run():
    # ensure graceful exit on SIGTERM as well
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    track_and_send()

if __name__ == "__main__":
    run()
