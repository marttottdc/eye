import os, time, base64, cv2, requests, numpy as np
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────────
WEIGHTS       = os.getenv("YOLO_WEIGHTS",  "yolov8n-face-lindevs.pt")
WEBHOOK_URL   = os.getenv("WEBHOOK_URL",   "https://developer.moio.ai/webhooks/f8284f2f-db74-4b45-a1e1-30479ac3117c/")
BEARER_TOKEN  = os.getenv("WEBHOOK_TOKEN", "e64465fbe22356fd66b429749a85c4783eb1262f5c763145ba4cb24585eb84f6")
VISUALIZE     = bool(int(os.getenv("VISUALIZE", "0")))
MIN_CONF      = float(os.getenv("MIN_CONF",  "0.40"))
SEND_COOLDOWN = float(os.getenv("SEND_COOLDOWN", "20.0"))
# ───────────────────────────────────────────────────────────────────


def encode_crop(img) -> str | None:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode("ascii") if ok else None


def search_person(b64_face: str, meta: dict) -> None:
    """Send the face + metadata to the webhook via POST/Bearer."""
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {"image_b64": b64_face, "meta": meta}

    try:
        resp = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=3)
        resp.raise_for_status()
        print(f"[INFO] sent track {meta['id']}  status={resp.status_code} response={resp.text}")
    except requests.RequestException as e:
        print(f"[ERR] webhook POST failed: {e}")

def main() -> None:
    model = YOLO(WEIGHTS)
    cap   = cv2.VideoCapture(0)
    last_sent: dict[int, float] = {}     # track_id → last-timestamp

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.track(frame, persist=True, conf=MIN_CONF)[0]

        # ── safe extraction of IDs (may be None the first frame) ──
        id_tensor = getattr(res.boxes, "id", None)
        if id_tensor is None:
            ids = np.full(len(res.boxes.xyxy), -1, dtype=int)
        else:
            ids = id_tensor.cpu().numpy().astype(int)
        # ──────────────────────────────────────────────────────────

        ts = time.time()

        for box, conf, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy(),
                                  ids):

            if tid == -1:                  # tracker hasn’t assigned anything
                continue
            if ts - last_sent.get(tid, 0) < SEND_COOLDOWN:
                continue                  # duplicate within cool-down

            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            b64  = encode_crop(crop)
            if not b64:
                continue

            meta = {
                "id":   int(tid),
                "conf": float(conf),
                "box":  [x1, y1, x2, y2],
                "ts":   ts,
            }
            search_person(b64, meta)
            last_sent[tid] = ts

        # optional live window
        if VISUALIZE:
            cv2.imshow("YOLOv8-Face (tracked)", res.plot())
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
        else:
            # allow graceful Ctrl-C / key quit even headless
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    if VISUALIZE:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
