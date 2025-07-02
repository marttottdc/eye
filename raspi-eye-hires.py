#!/usr/bin/env python3
import cv2, time, base64, requests, os
from dotenv import load_dotenv
from picamera2 import Picamera2

# Load .env config
load_dotenv()

# ---------- CONFIG -----------------------------------------------------------
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
STILL_WIDTH  = int(os.getenv("STILL_WIDTH", 1024))
STILL_HEIGHT = int(os.getenv("STILL_HEIGHT", 768))
MAX_CROP_SIZE = int(os.getenv("MAX_CROP_SIZE", 400))

SEND_EVERY   = int(os.getenv("SEND_EVERY", 20))
MAX_MISS     = int(os.getenv("MAX_MISS", 15))
DETECT_EVERY = int(os.getenv("DETECT_EVERY", 5))

WEBHOOK_URL   = os.getenv("WEBHOOK_URL")
BEARER_TOKEN  = os.getenv("WEBHOOK_TOKEN")

VISUALIZER = int(os.getenv("VISUALIZER", 0))  # 1 = ON, 0 = OFF
# -----------------------------------------------------------------------------

def send(face_id: int, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        print(f"[{face_id}] JPEG encoding failed")
        return
    b64 = base64.b64encode(buf).decode()
    try:
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json",
        }
        requests.post(WEBHOOK_URL, headers=headers, json={"id": face_id, "img_b64": b64}, timeout=10)
        print(f"[{face_id}] sent")
    except Exception as e:
        print(f"[{face_id}] send failed – {e}")

class FaceTracker:
    def __init__(self, max_miss=15):
        self.next_id = 0
        self.tracks  = {}
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

# Camera setup
cascade = cv2.CascadeClassifier(CASCADE_PATH)
if cascade.empty():
    raise IOError("Haar cascade XML not found!")

cam = Picamera2()

# Preview for detection
preview_config = cam.create_preview_configuration(
    main={"format": "XRGB8888", "size": (320, 240)})

# Still capture configuration
still_config = cam.create_still_configuration(
    main={"format": "XRGB8888", "size": (STILL_WIDTH, STILL_HEIGHT)})

cam.configure(preview_config)
cam.start()
time.sleep(0.5)

tracker = FaceTracker(max_miss=MAX_MISS)
frame_counter = 0

while True:
    frame = cam.capture_array()
    frame_counter += 1

    if frame_counter % DETECT_EVERY == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        tracks = tracker.update(faces)

        for fid, t in tracks.items():
            x, y, w, h = t["bbox"]

            if time.time() - t["last_sent"] >= SEND_EVERY:
                # Switch to still mode and capture high-res image
                cam.switch_mode_and_capture_file(still_config, "/tmp/highres.jpg")
                high_res_frame = cv2.imread("/tmp/highres.jpg")

                preview_width = 320
                highres_width = high_res_frame.shape[1]
                scale_factor = highres_width / preview_width

                hx = int(x * scale_factor)
                hy = int(y * scale_factor)
                hw = int(w * scale_factor)
                hh = int(h * scale_factor)

                hx = max(0, hx)
                hy = max(0, hy)
                hw = min(hw, high_res_frame.shape[1] - hx)
                hh = min(hh, high_res_frame.shape[0] - hy)

                crop = high_res_frame[hy:hy + hh, hx:hx + hw]

                if max(hw, hh) > MAX_CROP_SIZE:
                    crop = cv2.resize(crop, (MAX_CROP_SIZE, MAX_CROP_SIZE))

                send(fid, crop)
                t["last_sent"] = time.time()

                cam.configure(preview_config)
                cam.start()
                time.sleep(0.2)


    if VISUALIZER and frame_counter % 10 == 0:
        for fid, t in tracker.tracks.items():
            x, y, w, h = t["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {fid}", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow("Face-Tracker", frame)

    if VISUALIZER:
        if cv2.waitKey(1) == 27:
            break
    else:
        time.sleep(0.01)

cv2.destroyAllWindows()
