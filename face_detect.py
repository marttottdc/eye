from ultralytics import YOLO
import cv2, base64, time

# ── CONFIG ─────────────────────────────────────────────
WEIGHTS        = "yolov8x-face-lindevs.pt"   # 👈 pre-trained face detector
VISUALIZE      = False               # set True for live preview
MIN_CONF       = 0.7                 # confidence filter
MAX_FPS        = 30                  # optional throttle
# ──────────────────────────────────────────────────────

def search_person(b64_face, meta):
    """Your lookup logic goes here."""
    print(f"[{meta['ts']:.2f}] face sent ({meta['box']}) conf={meta['conf']:.2f}")


def encode_crop(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode("ascii") if ok else None


def main():
    model = YOLO(WEIGHTS)
    cap   = cv2.VideoCapture(0)
    last  = 0.0

    while True:
        ret, frame = cap.read()

        res = model.track(frame,
                          persist=True,  # keep IDs across frames
                          conf=MIN_CONF)[0]

        ts = time.time()

        if not ret:
            break

        # FPS cap
        if MAX_FPS and time.time() - last < 1.0 / MAX_FPS:
            continue
        last = time.time()

        # Face detection (no class filter needed – model only knows “face”)
        res = model(frame, conf=MIN_CONF)[0]

        for box, conf in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            b64  = encode_crop(crop)
            if b64:
                meta = {"ts": time.time(), "conf": float(conf),
                        "box": [x1, y1, x2, y2]}
                search_person(b64, meta)

        if VISUALIZE:
            cv2.imshow("YOLOv8 – faces", res.plot())
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
        else:
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    if VISUALIZE:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
