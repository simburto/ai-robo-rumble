import cv2
import numpy as np
from multiprocessing import *
import time
from mss import mss
import keyboard
import pyKey

q = Queue()

result = {
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
    'counter': 0,
}
q.put(result,)

def vision():
        while True:
            result = q.get()
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            sct = mss()
            while True:
                start = time.time()
                frame = np.array(sct.grab(monitor))
                result['frame'] = frame
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                result['hsv'] = hsv
                q.put(result,)
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
def red_balls():
    result = q.get()
    if np.any(result['hsv']) and result['counter'] != 4:
        if np.any(result['hsv']):
            hsv = result['hsv']
            lower_red = np.array([1, 202, 236])
            upper_red = np.array([3, 204, 238])
            ball_mask = cv2.inRange(hsv, lower_red, upper_red)
            edges = cv2.Canny(ball_mask,100,200)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=0, maxRadius=35)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                result['red_ball_count'] = len(circles)
                result['red_ball_box'] = circles.tolist()
                result['red_ball_pos'] = [(x, y) for (x, y, _) in circles]
                q.put(result,)

if __name__ == '__main__':
    visionthread = Process(target=vision)
    visionthread.start()
    redballthread = Process(target=red_balls)
    redballthread.start()
    while True:
        time.sleep(1)
