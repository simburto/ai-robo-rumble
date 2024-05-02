import cv2
import numpy as np
import multiprocessing
import time
from mss import mss
import keyboard
import pyKey
if __name__ == '__main__':
    result = multiprocessing.Manager().dict({
        'robot_center': None,
        'intake_angle': None,
        'red_ball_count': 0,
        'red_ball_pos': [],
        'cargo_count': 0,
        'blue_ball_count': 0,
        'blue_ball_pos': [],
        'hsv': None,
        'frame': None,
        'bumper_box': None,
        'intake_box': None,
        'intake_side': None,
        'red_ball_box': None,
        'climb_box': None,
        'blue_ball_box': None,
        'time': None,
    })

def vision(result):
        while True:
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            sct = mss()
            while True:
                start = time.time()
                frame = np.array(sct.grab(monitor))
                result['frame'] = frame
                print(frame, frame.dtype)
                print(type(frame))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                result['hsv'] = hsv
                if np.any(hsv) and keyboard.is_pressed('q'):
                    lower_climb = np.array([53,184,182]) 
                    upper_climb = np.array([55,186,184])

                    climb_mask = cv2.inRange(hsv, lower_climb, upper_climb)

                    contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 1:
                        pyKey.press(key='r',sec=0.1)
                cv2.waitKey(1)
                cv2.imshow('vision', frame)
                print(1/(time.time() - start))

if __name__ == '__main__':
    visionthread = multiprocessing.Process(target=vision, args=(result,))
    visionthread.start()
    while True:
        time.sleep(1)