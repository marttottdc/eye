import cv2, time
from picamera2 import Picamera2

CASCADE = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
        main={"format": "XRGB8888", "size": (320, 240)}))  # 320×240 keeps CPU <75 %
cam.start()
time.sleep(0.5)  # give AGC a moment

while True:
    frame = cam.capture_array()          # ~35 fps raw → we’ll drop some
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.15, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Pi-3 FaceCam", frame)
    if cv2.waitKey(1) == 27:             # Esc to quit
        break
