import cv2

cap = cv2.VideoCapture(0)

# Load the Daimler frontal-only detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(4, 4),     # smaller stride → finer search
        padding=(8, 8),       # less padding if you know people never touch frame edges
        scale=1.02            # tiny scale steps because size won’t vary much
    )

    if rects.any():
        print("👤 Person detected head-on!")
    else:
        print("– no one in view")

    # draw
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Head-On HOG", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
