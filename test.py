import cv2
import numpy as np
import math
import threading
import pyKey
import time
from mss import mss

def vision():
        status = False
        while True:
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            sct = mss()
            if status == False:
                print("Vision thread started.")
                status = True
            frame = np.array(sct.grab(monitor))
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            cv2.waitKey(1)
            cv2.imshow('vision', frame)

vision_thread = threading.Thread(target=vision)
vision_thread.start()

while True:
     time.sleep(1)