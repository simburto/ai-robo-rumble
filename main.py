import vision
import cv2
import pyKey
import time

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 1280,720)

timestarted = False

def autoclimb(needlebboxcount):
    if needlebboxcount == 2:
        pyKey.press(key='r',sec=0.1)

for result in vision.vision():
    autoclimb(result['climb_counter'])
    frame = result.get('frame', None)
    cv2.imshow("Live", frame)
    if ['endgame'] == False:
        timestarted = False
    else:
        if timestarted == False:
            start = time.time()
            timestarted = True
        print(f"{time.time() - start} , {time.time()},{start}")
    if time.time() - start > 33 and timestarted == True:
        print("game end")
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()