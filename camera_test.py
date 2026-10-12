import cv2

print("OpenCV Version:", cv2.__version__)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

print("Opened:", cap.isOpened())

if cap.isOpened():
    ret, frame = cap.read()
    print("Read:", ret)

cap.release()