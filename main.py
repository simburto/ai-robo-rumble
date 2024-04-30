import vision
import cv2
import pyKey
import time

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 1280,720)

timestarted = False

def autoclimb(needlebboxcount):
    if needlebboxcount > 1:
        pyKey.press(key='r',sec=0.1)

for result in vision.vision():
    if result['climb_counter']:
        autoclimb(result['climb_counter'])
    frame = result.get('frame', None)
    cv2.imshow("Live", frame)
    if timestarted == False:
        start = time.time()
    if result['balls_coords']:
        timestarted = True

    print(f"{time.time() - start} , {time.time()},{start}")
    if time.time() - start > 155 and timestarted == True:
        break
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()