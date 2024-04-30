import vision
import cv2
cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 1280,720)
while True:
    frame = vision.vision()
    print(frame)