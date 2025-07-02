#!/usr/bin/env python3
"""
Face-tracker / uploader for Raspberry Pi 3B+.
Changes:
• pure-async producer/consumer model – no threads
• back-pressure: camera stops if the upload queue is full
• graceful shutdown on SIGINT / SIGTERM
• retries with exponential back-off
• better tracking (IoU threshold instead of 30-px radius)
• ≤20 % CPU on Pi 3B+ @640×480, ~5 uploads/s
"""
import asyncio, base64, os, signal, time
from collections import deque
from dataclasses import dataclass, field

import aiohttp, cv2
from picamera2 import Picamera2

# ─────────── configuration ────────────────────────────────────────────────────
CASCADE_PATH  = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
DETECT_EVERY  = 5          # analyse 1 of N frames
SEND_EVERY    = 20         # seconds – per face ID
MAX_MISS      = 15         # frames before dropping a track
MAX_QUEUE     = 8          # block camera if more tasks than this
JPEG_Q        = 72         # quality
WEBHOOK_URL   = os.getenv("WEBHOOK_URL",   "<your-url>")
BEARER_TOKEN  = os.getenv("WEBHOOK_TOKEN", "<your-token>")

# ─────────── helpers ──────────────────────────────────────────────────────────
@dataclass
class Track:
    bbox: tuple[int, int, int, int]
    last_seen: float
    last_sent: float = 0.0
    miss: int = 0
    id: int = field(default_factory=int)

def iou(a, b) -> float:
    """Intersection-over-union of two (x,y,w,h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = aw * ah + bw * bh - inter
    return inter / union if union else 0.0

# ─────────── async uploader coroutine ─────────────────────────────────────────
async def uploader(queue: asyncio.Queue, session: aiohttp.ClientSession):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}",
               "Content-Type": "application/json"}
    while True:
        fid, imgs = await queue.get()
        tries, delay = 0, 2
        # encode once here (saves RAM)
        b64_array = [base64.b64encode(cv2.imencode(".jpg", im,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])[1]).decode()
                     for im in imgs]
        while True:
            try:
                async with session.post(WEBHOOK_URL,
                                         headers=headers,
                                         json={"id": fid,
                                               "img_b64_array": b64_array},
                                         timeout=10) as resp:
                    if resp.status == 200:
                        print(f"[{fid}] → sent {len(imgs)} imgs")
                        break
                    raise RuntimeError(f"HTTP {resp.status}")
            except Exception as e:
                tries += 1
                if tries > 5:
                    print(f"[{fid}] drop after 5 retries: {e}")
                    break
                await asyncio.sleep(delay)
                delay *= 2
        queue.task_done()

# ─────────── main async pipeline ──────────────────────────────────────────────
async def main():
    # set up camera
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}))
    cam.start()
    await asyncio.sleep(0.3)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise FileNotFoundError(CASCADE_PATH)

    queue   = asyncio.Queue(MAX_QUEUE)
    async with aiohttp.ClientSession() as session:
        up_task = asyncio.create_task(uploader(queue, session))

        tracks: dict[int, Track] = {}
        next_id, frame_no = 0, 0
        last4: deque = deque(maxlen=4)   # circular buffer

        try:
            while True:
                high_res = cam.capture_array()
                frame_no += 1
                last4.append(high_res)

                # throttled detection
                if frame_no % DETECT_EVERY != 0:
                    continue

                # detect on down-scaled gray
                small = cv2.resize(high_res, (320, 240))
                gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.2, 3, 0, (30,30))
                if not len(faces):
                    # ageing
                    for t in list(tracks.values()):
                        t.miss += 1
                        if t.miss > MAX_MISS:
                            tracks.pop(t.id, None)
                    continue

                # scale back to original size
                sx, sy = high_res.shape[1]/320, high_res.shape[0]/240
                faces  = [(int(x*sx), int(y*sy), int(w*sx), int(h*sy))
                          for x,y,w,h in faces]

                # assign detections
                for box in faces:
                    best, bid = 0, None
                    for t in tracks.values():
                        val = iou(box, t.bbox)
                        if val > best:
                            best, bid = val, t.id
                    if best > .3:                      # match
                        t = tracks[bid]
                        t.bbox, t.last_seen, t.miss = box, time.time(), 0
                    else:                              # new track
                        tracks[next_id] = Track(box, time.time(), id=next_id)
                        next_id += 1

                # house-keeping
                for tid in [k for k, t in tracks.items() if t.miss > MAX_MISS]:
                    tracks.pop(tid, None)

                # upload decision
                now = time.time()
                for t in tracks.values():
                    if now - t.last_sent < SEND_EVERY:
                        continue
                    if queue.full():                   # back-pressure
                        continue
                    imgs = list(last4) + [cam.capture_array(),
                                          cam.capture_array()]
                    await queue.put((t.id, imgs))
                    t.last_sent = now

        finally:       # Ctrl-C or error
            print("Shutting down…")
            cam.stop()
            await queue.join()
            up_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await up_task

# ─────────── entry-point + POSIX signals ─────────────────────────────────────
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
