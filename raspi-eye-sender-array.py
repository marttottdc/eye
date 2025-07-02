#!/usr/bin/env python3
"""
Async face-tracker/uploader for Raspberry Pi 3 B+.
– Haar-cascade detection every N frames on a down-scaled copy
– Track faces with IoU, assign stable IDs
– Upload five JPEGs per ID (n-2…n+2) to a webhook with retries
– Console logging + heartbeat
"""

import asyncio, base64, contextlib, logging, os, signal, time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple, List

import aiohttp, cv2
from picamera2 import Picamera2

# ───────────────────────────── config ─────────────────────────────────────────
CASCADE_PATH   = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
DETECT_EVERY   = 5           # analyse 1 out of N frames
SEND_EVERY     = 20          # seconds between uploads per face ID
MAX_MISS       = 15          # frames before a track is dropped
MAX_QUEUE      = 8           # back-pressure – block producer if > N batches pending
JPEG_Q         = 72          # jpeg quality
HEARTBEAT_SEC  = 30          # heartbeat interval

WEBHOOK_URL    = os.getenv("WEBHOOK_URL",   "https://developer.moio.ai/webhooks/f8284f2f-db74-4b45-a1e1-30479ac3117c/")
BEARER_TOKEN   = os.getenv("WEBHOOK_TOKEN", "e64465fbe22356fd66b429749a85c4783eb1262f5c763145ba4cb24585eb84f6")

# ──────────────────────────── logging ─────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("tracker")

log = setup_logging()

# ────────────────────────── dataclasses ───────────────────────────────────────
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
    return [
        base64.b64encode(
            cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])[1]
        ).decode()
        for im in imgs
    ]

# ─────────────────────── async coroutines ─────────────────────────────────────
async def uploader(q: asyncio.Queue):
    """Background consumer pushing batches to WEBHOOK_URL."""
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}",
               "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        while True:
            fid, imgs = await q.get()
            b64_array = encode_images(imgs)

            tries, delay = 0, 2
            while True:
                try:
                    async with session.post(
                        WEBHOOK_URL,
                        headers=headers,
                        json={"id": fid, "img_b64_array": b64_array},
                        timeout=10,
                    ) as resp:
                        if resp.status == 200:
                            log.info(f"[{fid}] sent {len(imgs)} imgs")
                            break
                        raise RuntimeError(f"HTTP {resp.status}")
                except Exception as e:
                    tries += 1
                    if tries > 5:
                        log.error(f"[{fid}] drop after 5 retries: {e}")
                        break
                    await asyncio.sleep(delay)
                    delay *= 2
            q.task_done()

async def heartbeat(q: asyncio.Queue, tracks: Dict[int, Track]):
    """Periodic status message."""
    while True:
        await asyncio.sleep(HEARTBEAT_SEC)
        log.debug(f"♥ heartbeat – queue={q.qsize()} tracks={len(tracks)}")

# ──────────────────────────── main loop ───────────────────────────────────────
async def track_and_send():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise FileNotFoundError(CASCADE_PATH)

    cam             = create_camera()
    frame_no        = 0
    last4: Deque    = deque(maxlen=4)
    tracks: Dict[int, Track] = {}
    next_id         = 0

    queue = asyncio.Queue(MAX_QUEUE)
    up_task  = asyncio.create_task(uploader(queue))
    hb_task  = asyncio.create_task(heartbeat(queue, tracks))

    try:
        while True:
            high_res = cam.capture_array()
            frame_no += 1
            last4.append(high_res)

            # skip unless this is a detection frame
            if frame_no % DETECT_EVERY:
                continue

            # run cascade on down-scaled frame
            small  = cv2.resize(high_res, (320, 240))
            gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces  = cascade.detectMultiScale(gray, 1.2, 3, 0, (30, 30))
            sx, sy = high_res.shape[1] / 320, high_res.shape[0] / 240
            faces  = [(int(x * sx), int(y * sy), int(w * sx), int(h * sy))
                      for x, y, w, h in faces]

            # age existing tracks
            for t in tracks.values():
                t.miss += 1

            # assign detections → tracks
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

            # drop stale tracks
            for tid in [tid for tid, tr in tracks.items() if tr.miss > MAX_MISS]:
                tracks.pop(tid, None)

            # enqueue uploads
            now = time.time()
            for tr in tracks.values():
                if now - tr.last_sent < SEND_EVERY:
                    continue
                if queue.full():
                    continue
                imgs = list(last4) + [cam.capture_array(), cam.capture_array()]
                await queue.put((tr.id, imgs))
                log.info(f"SEND  ↥ queued ID {tr.id} ({len(imgs)} imgs)")
                tr.last_sent = now
    finally:
        log.info("Shutting down…")
        cam.stop()
        await queue.join()
        up_task.cancel(), hb_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await up_task
            await hb_task

# ────────────────────────── entry-point ───────────────────────────────────────
def run():
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)
    try:
        loop.run_until_complete(track_and_send())
    finally:
        loop.close()

if __name__ == "__main__":
    run()
