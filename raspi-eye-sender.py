#!/usr/bin/env python3
import cv2, time, base64, requests, os
from picamera2 import Picamera2

# ---------- CONFIG -----------------------------------------------------------
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
SEND_EVERY   = 10          # seconds between uploads per ID
MAX_MISS     = 15          # frames to wait before dropping an ID

WEBHOOK_URL   = os.getenv("WEBHOOK_URL",   "https://developer.moio.ai/webhooks/f8284f2f-db74-4b45-a1e1-30479ac3117c/")
BEARER_TOKEN  = os.getenv("WEBHOOK_TOKEN", "e64465fbe22356fd66b429749a85c4783eb1262f5c763145ba4cb24585eb84f6")
# -----------------------------------------------------------------------------

# helper ──────────────────────────────────────────────────────────────────────
def send(face_id: int, img):
    """Encode img as JPEG → base64 → POST.  Adapt at will."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return
    b64 = base64.b64encode(buf).decode()
    try:
        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json",
        }
        requests.post(WEBHOOK_URL, headers=headers, json={"id": face_id, "img_b64": b64}, timeout=30)
        print(f"[{face_id}] sent")
    except Exception as e:
        print(f"[{face_id}] send failed – {e}")

# very-light centroid tracker ────────────────────────────────────────────────
class FaceTracker:
    def __init__(self, max_miss=15):
        self.next_id = 0
        self.tracks  = {}   # id -> dict(bbox, last_seen, last_sent, miss)
        self.max_miss = max_miss

    @staticmethod
    def _center(b):  # bbox (x,y,w,h) → (cx,cy)
        return (b[0]+b[2]//2, b[1]+b[3]//2)

    def update(self, detections):
        now = time.time()
        # mark all as missed once; we’ll clear miss if matched
        for t in self.tracks.values():
            t["miss"] += 1

        # naive matching: nearest centre <= 40 px
        for det in detections:
            cx, cy = self._center(det)
            best_id, best_d = None, 1e9
            for fid, t in self.tracks.items():
                tx, ty = self._center(t["bbox"])
                d = (cx-tx)**2 + (cy-ty)**2
                if d < best_d and d < 40**2:
                    best_id, best_d = fid, d
            if best_id is None:                 # new face
                fid = self.next_id; self.next_id += 1
                self.tracks[fid] = dict(bbox=det, last_seen=now,
                                        last_sent=0, miss=0)
            else:                               # existing face
                t = self.tracks[best_id]
                t.update(bbox=det, last_seen=now, miss=0)

        # purge stale IDs
        gone = [fid for fid, t in self.tracks.items() if t["miss"] > self.max_miss]
        for fid in gone:
            del self.tracks[fid]

        return self.tracks

# init ────────────────────────────────────────────────────────────────────────
cascade = cv2.CascadeClassifier(CASCADE_PATH)
if cascade.empty():
    raise IOError("Haar cascade XML not found!")

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"format": "XRGB8888", "size": (320, 240)}))
cam.start()
time.sleep(0.5)

tracker = FaceTracker(max_miss=MAX_MISS)

# loop ────────────────────────────────────────────────────────────────────────
while True:
    frame = cam.capture_array()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.15, 5)

    tracks = tracker.update(faces)

    for fid, t in tracks.items():
        x, y, w, h = t["bbox"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"ID {fid}", (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        # if 10 s have passed → send
        if time.time() - t["last_sent"] >= SEND_EVERY:
            crop = frame[y:y+h, x:x+w]
            send(fid, crop)
            t["last_sent"] = time.time()

    cv2.imshow("Face-Tracker", frame)
    if cv2.waitKey(1) == 27:      # Esc to quit
        break

cv2.destroyAllWindows()
