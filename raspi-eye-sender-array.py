#!/usr/bin/env python3
import cv2, time, base64, asyncio, aiohttp, os
from picamera2 import Picamera2

# ---------- CONFIG -----------------------------------------------------------
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
SEND_EVERY = 20  # seconds between uploads per ID
MAX_MISS = 15  # frames to wait before dropping an ID
DETECT_EVERY = 5  # detect every N frames to reduce CPU load

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://developer.moio.ai/webhooks/f8284f2f-db74-4b45-a1e1-30479ac3117c/")
BEARER_TOKEN = os.getenv("WEBHOOK_TOKEN", "e64465fbe22356fd66b429749a85c4783eb1262f5c763145ba4cb24585eb84f6")


# -----------------------------------------------------------------------------

async def send(face_id: int, img_array, session: aiohttp.ClientSession):
    b64_array = []
    for img in img_array:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            print(f"[{face_id}] JPEG encoding failed for one image")
            continue
        b64 = base64.b64encode(buf).decode()
        b64_array.append(b64)

    if not b64_array:
        print(f"[{face_id}] No valid images to send")
        return

    try:
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {"id": face_id, "img_b64_array": b64_array}
        async with session.post(WEBHOOK_URL, headers=headers, json=payload, timeout=10) as response:
            if response.status == 200:
                print(f"[{face_id}] sent {len(b64_array)} images")
            else:
                print(f"[{face_id}] send failed – HTTP {response.status}")
    except Exception as e:
        print(f"[{face_id}] send failed – {e}")


class FaceTracker:
    def __init__(self, max_miss=15):
        self.next_id = 0
        self.tracks = {}
        self.max_miss = max_miss

    @staticmethod
    def _center(b):
        return (b[0] + b[2] // 2, b[1] + b[3] // 2)

    def update(self, detections):
        now = time.time()
        for t in self.tracks.values():
            t["miss"] += 1

        for det in detections:
            cx, cy = self._center(det)
            best_id, best_d = None, 1e9
            for fid, t in self.tracks.items():
                tx, ty = self._center(t["bbox"])
                d = (cx - tx) ** 2 + (cy - ty) ** 2
                if d < best_d and d < 30 ** 2:
                    best_id, best_d = fid, d
            if best_id is None:
                fid = self.next_id
                self.next_id += 1
                self.tracks[fid] = dict(bbox=det, last_seen=now, last_sent=0, miss=0)
            else:
                t = self.tracks[best_id]
                t.update(bbox=det, last_seen=now, miss=0)

        gone = [fid for fid, t in self.tracks.items() if t["miss"] > self.max_miss]
        for fid in gone:
            del self.tracks[fid]

        return self.tracks


# init
cascade = cv2.CascadeClassifier(CASCADE_PATH)
if cascade.empty():
    raise IOError("Haar cascade XML not found!")

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"format": "XRGB8888", "size": (640, 480)}))  # Capture at 640x480
cam.start()
time.sleep(0.5)

tracker = FaceTracker(max_miss=MAX_MISS)
frame_counter = 0
frame_buffer = []  # Store up to 4 frames (n-4 to n-1)

# Async setup
loop = asyncio.get_event_loop()
session = aiohttp.ClientSession()

try:
    while True:
        # Capture high-res frame
        high_res = cam.capture_array()
        frame_counter += 1

        if frame_counter % DETECT_EVERY == 0:
            # Detect on current frame (n)
            detect_size = (320, 240)
            frame_n = cv2.resize(high_res, detect_size)
            gray_n = cv2.cvtColor(frame_n, cv2.COLOR_BGR2GRAY)
            faces_n = cascade.detectMultiScale(gray_n, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

            # Scale detections to high-res
            scale_x = high_res.shape[1] / detect_size[0]
            scale_y = high_res.shape[0] / detect_size[1]
            scaled_faces_n = [(int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)) for (x, y, w, h)
                              in faces_n]

            tracks = tracker.update(scaled_faces_n)

            for fid, t in tracks.items():
                if time.time() - t["last_sent"] >= SEND_EVERY:
                    # Collect n-2, n-1, n, n+1, n+2 (full frames)
                    img_array = []

                    # Get n-2 and n-1 from buffer
                    if len(frame_buffer) >= 2:
                        img_array.append(frame_buffer[-2])  # n-2
                    if len(frame_buffer) >= 1:
                        img_array.append(frame_buffer[-1])  # n-1

                    # Add n (current frame)
                    img_array.append(high_res)

                    # Capture n+1 and n+2
                    n1_plus = cam.capture_array()
                    img_array.append(n1_plus)
                    n2_plus = cam.capture_array()
                    img_array.append(n2_plus)

                    # Schedule async send
                    if img_array:
                        asyncio.create_task(send(fid, img_array, session))
                        t["last_sent"] = time.time()

        # Update frame buffer (keep last 4 frames)
        frame_buffer.append(high_res.copy())
        if len(frame_buffer) > 4:
            frame_buffer.pop(0)

        # Display (optional, commented out for performance)
        """
        if frame_counter % 10 == 0:
            display_frame = cv2.resize(high_res, (320, 240))
            for fid, t in tracker.tracks.items():
                x, y, w, h = [int(coord * 320 / high_res.shape[1]) for coord in t["bbox"]]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID {fid}", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow("Face-Tracker", display_frame)
        """

        if cv2.waitKey(1) == 27:
            break

finally:
    cam.stop()
    # Clean up async session
    loop.run_until_complete(session.close())
    cv2.destroyAllWindows()