import vision
import cv2

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 1280,720)

for result in vision.vision():
    frame = result.get('frame', None)
    cv2.imshow("Live", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()