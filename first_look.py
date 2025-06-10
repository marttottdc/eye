import cv2
import datetime

# open camera
cap = cv2.VideoCapture(0)

# setup HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect people
        boxes, _ = hog.detectMultiScale(frame,
                                         winStride=(8, 8),
                                         padding=(16, 16),
                                         scale=1.05)

        # print a simple flag: 1 = someone in view, 0 = none

        if len(boxes) > 0:
            now = datetime.datetime.now().isoformat(timespec='seconds')
            print(f'Persona detected {now}')

        # if you need a small delay to throttle CPU:
        # time.sleep(0.05)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
