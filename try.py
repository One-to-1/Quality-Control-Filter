import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()