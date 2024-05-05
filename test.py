import numpy as np
import time
import keyboard
import cv2
import pyKey
import sys
from mss import mss
import multiprocessing
h = multiprocessing.SimpleQueue()
f = multiprocessing.SimpleQueue()
def vision():
        lower_climb = np.array([53,184,182]) 
        upper_climb = np.array([55,186,184])
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        while True:
            with mss() as sct:
                start = time.time()
                frame = np.array(sct.grab(monitor))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                f.put(frame)
                h.put(hsv)
                h.put(hsv)
                h.put(hsv)
                h.put(hsv)
                if np.any(hsv) and keyboard.is_pressed('q'):
                    climb_mask = cv2.inRange(hsv, lower_climb, upper_climb)
                    contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 1:
                        pyKey.press(key='r',sec=0.1)
                sys.stdout.write(f"\r{1/(time.time() - start)}")
                sys.stdout.flush()
                sct.close()

vision()

while True:
    h.get_nowait()
    f.get_nowait()
    time.sleep(0.001)