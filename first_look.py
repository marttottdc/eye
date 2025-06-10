import cv2

# 1. Open the camera
cap = cv2.VideoCapture(0)

# 2. Initialize the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Detect people
    #    winStride can be tuned for speed vs accuracy
    boxes, weights = hog.detectMultiScale(frame,
                                          winStride=(8,8),
                                          padding=(16,16),
                                          scale=1.05)
    if len(boxes) > 0:
        print("👤 Person detected!")
    else:
        print("– no person")

    # 4. (Optional) draw boxes and show
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Person Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
