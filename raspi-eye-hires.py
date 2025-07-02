#!/usr/bin/env python3
import cv2, time, base64, requests, os, sys
from dotenv import load_dotenv
from picamera2 import Picamera2

load_dotenv()

# ---------- CONFIG ----------------------------------------------------------
CASCADE_PATH   = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
STILL_WIDTH    = int(os.getenv("STILL_WIDTH", 1024))
STILL_HEIGHT   = int(os.getenv("STILL_HEIGHT", 768))

SEND_EVERY     = int(os.getenv("SEND_EVERY", 20))
MAX_MISS       = int(os.getenv("MAX_MISS", 15))
DETECT_EVERY   = int(os.getenv("DETECT_EVERY", 5))
VISUALIZER     = int(os.getenv("VISUALIZER", 0))

WEBHOOK_URL    = os.getenv("WEBHOOK_URL")
BEARER_TOKEN   = os.getenv("WEBHOOK_TOKEN")
if not (WEBHOOK_URL and BEARER_TOKEN):
    sys.exit("WEBHOOK_URL / WEBHOOK_TOKEN missing in environment")

# ---------- HELPERS ---------------------------------------------------------
def send(face_id: int, img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        print(f"[{face_id}] JPEG encode failed")
        return
    try:
        requests.post(
            WEBHOOK_URL,
            headers={"Authorization": f"Bearer {BEARER_TOKEN}",
                     "Content-Type": "application/json"},
            json={"id": face_id, "img_b64": base64.b64encode(buf).decode()},
            timeout=10,
        )
        print(f"[{face_id}] sent")
    except Exception as e:
        print(f"[{face_id}] send failed – {e}")

class FaceTracker:
    def __init__(self, max_miss=15):
        self.next_id = 0
        self.tracks  = {}
        self.max_miss = max_miss
    @staticmethod
    def _center(b): return (b[0]+b[2]//2, b[1]+b[3]//2)
    def update(self, detections):
        for t in self.tracks.values(): t["miss"] += 1
        for det in detections:
            cx,cy = self._center(det); best_id,best_d=None,1e9
            for fid,t in self.tracks.items():
                tx,ty = self._center(t["bbox"])
                d=(cx-tx)**2+(cy-ty)**2
                if d<best_d and d<30**2: best_id,best_d=fid,d
            if best_id is None:
                fid=self.next_id; self.next_id+=1
                self.tracks[fid]=dict(bbox=det,last_sent=0,miss=0)
            else:
                self.tracks[best_id].update(bbox=det,miss=0)
        self.tracks={fid:t for fid,t in self.tracks.items()
                     if t["miss"]<=self.max_miss}
        return self.tracks

# ---------- CAMERA INIT -----------------------------------------------------
cascade=cv2.CascadeClassifier(CASCADE_PATH)
if cascade.empty(): sys.exit("Cascade not found")

cam=Picamera2()
preview_cfg=cam.create_preview_configuration(main={"format":"XRGB8888","size":(320,240)})
still_cfg  =cam.create_still_configuration(main={"format":"XRGB8888","size":(STILL_WIDTH,STILL_HEIGHT)})

cam.configure(preview_cfg)
cam.start(); time.sleep(0.3)

tracker, frame_ctr = FaceTracker(MAX_MISS), 0

# ---------- MAIN LOOP -------------------------------------------------------
try:
    while True:
        frame = cam.capture_array(); frame_ctr += 1

        if frame_ctr % DETECT_EVERY == 0:
            faces = cascade.detectMultiScale(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        1.1, 5, minSize=(30,30))
            tracks = tracker.update(faces)

            for fid,t in tracks.items():
                if time.time()-t["last_sent"] < SEND_EVERY: continue

                # capture full-res still
                cam.stop()
                cam.configure(still_cfg); cam.start(); time.sleep(0.25)
                hi_frame = cam.capture_array()

                cam.stop()
                cam.configure(preview_cfg); cam.start(); time.sleep(0.25)

                send(fid, hi_frame)
                t["last_sent"] = time.time()

        if VISUALIZER and frame_ctr % 10 == 0:
            for fid,t in tracker.tracks.items():
                x,y,w,h = t["bbox"]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,f"ID {fid}",(x,y-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
            cv2.imshow("Face-Tracker", frame)

        if VISUALIZER:
            if cv2.waitKey(1)==27: break
        else:
            time.sleep(0.01)

finally:
    cam.stop()
    if VISUALIZER: cv2.destroyAllWindows()
