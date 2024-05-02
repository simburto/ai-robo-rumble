import cv2
import numpy as np
import multiprocessing
import time
import keyboard
import pyKey
from mss import mss

def vision(shared_frame, shared_red_ball_pos, shared_hsv_ready, shared_counter):
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    sct = mss()
    while True:
        start = time.time()
        frame = np.array(sct.grab(monitor))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_shape = frame.shape
        frame = frame.flatten()
        shared_frame[:len(frame)] = frame
        shared_hsv_ready.value = 1
        if np.any(hsv) and keyboard.is_pressed('q'):
            lower_climb = np.array([53, 184, 182]) 
            upper_climb = np.array([55, 186, 184])
            climb_mask = cv2.inRange(hsv, lower_climb, upper_climb)
            contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                pyKey.press(key='r', sec=0.1)
        shared_counter.value += 1
        print(1 / (time.time() - start))

def red_balls(shared_frame, shared_hsv_ready, shared_red_ball_count, shared_red_ball_pos, shared_counter):
    while True:
        if shared_hsv_ready.value == 1 and shared_counter.value != 4:
            frame = np.frombuffer(shared_frame[:], dtype=np.uint8).reshape((1080, 1920, 3))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([1, 202, 236])
            upper_red = np.array([3, 204, 238])
            ball_mask = cv2.inRange(hsv, lower_red, upper_red)
            edges = cv2.Canny(ball_mask, 100, 200)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=0, maxRadius=35)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                shared_red_ball_count.value = len(circles)
                shared_red_ball_pos[:] = circles.flatten()


if __name__ == "__main__":
    shared_frame = multiprocessing.Array('B', 1920 * 1080 * 4)
    shared_red_ball_pos = multiprocessing.Array('i', 0)
    shared_hsv_ready = multiprocessing.Value('i', 0)
    shared_red_ball_count = multiprocessing.Value('i', 0)
    shared_counter = multiprocessing.Value('i', 0)

    vision_process = multiprocessing.Process(target=vision, args=(shared_frame, shared_red_ball_pos, shared_hsv_ready, shared_counter))
    red_balls_process = multiprocessing.Process(target=red_balls, args=(shared_hsv_ready, shared_red_ball_count, shared_red_ball_pos, shared_counter))

    vision_process.start()
    red_balls_process.start()

    while True:
        time.sleep(1)
