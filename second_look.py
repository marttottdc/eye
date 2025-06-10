#!/usr/bin/env python3
"""
person_detect_dnn.py

Run a live person detector using MobileNet-SSD + OpenCV’s DNN module.
Prints ISO-timestamps and a simple flag (1=person, 0=none). Optionally
draws boxes and shows a window if you remove `--headless`.
"""

import cv2
import time
import datetime
import argparse

# Class IDs in the SSD model; 15 == "person"
PERSON_CLASS_ID = 15


def main():
    parser = argparse.ArgumentParser(
        description="Headless person detection with MobileNet-SSD"
    )
    parser.add_argument(
        "--prototxt",
        required=True,
        help="Path to deploy.prototxt"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MobileNetSSD_deploy.caffemodel"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum probability to filter weak detections"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable any GUI output (no imshow)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture frame width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture frame height"
    )
    args = parser.parse_args()

    # Load the DNN model
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # Prepare blob for DNN: resize, normalize, swap BGR→RGB
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                scalefactor=0.007843,
                size=(300, 300),
                mean=(127.5, 127.5, 127.5),
                swapRB=False,  # model was trained with BGR
                crop=False
            )
            net.setInput(blob)
            detections = net.forward()

            person_seen = False
            boxes = []

            # Loop over detections
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                cls  = int(detections[0, 0, i, 1])
                if cls == PERSON_CLASS_ID and conf >= args.confidence:
                    person_seen = True
                    # Extract bounding box coords (normalized → pixel)
                    box = detections[0, 0, i, 3:7] * [
                        frame.shape[1], frame.shape[0],
                        frame.shape[1], frame.shape[0]
                    ]
                    (x1, y1, x2, y2) = box.astype("int")
                    boxes.append((x1, y1, x2, y2))

            # Timestamp and flag
            now = datetime.datetime.now().isoformat(timespec='seconds')
            flag = 1 if person_seen else 0
            print(f"{now} — person_in_sight={flag}")

            # If not headless, draw boxes & show
            if not args.headless:
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )
                cv2.imshow("Person Detector (DNN)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Throttle to ~10 FPS (adjust as needed)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n👋 Exiting on user interrupt")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
