import cv2
import numpy as np
import math
import pyKey
import time
import bettercam
import multiprocessing
import keyboard
import sys
from paddleocr import PaddleOCR

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
kernel = np.ones((3, 3), np.uint8)
ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)

h = multiprocessing.JoinableQueue()
scale = multiprocessing.JoinableQueue()
result = multiprocessing.JoinableQueue()
flag = multiprocessing.Event()
count = 0

class vision():
    def vision(h, scale, result, flag):
        lower_climb = np.array([53, 184, 182])
        upper_climb = np.array([55, 186, 184])
        camera = bettercam.create(output_color="BGR")
        while True:
            frame = np.array(camera.grab())
            if np.any(frame):
                start = time.time()
                frame = frame.astype(np.uint8)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                downscaled = cv2.resize(hsv, (1152, 648), interpolation=cv2.INTER_LINEAR)
                scale.put(hsv)
                h.put(downscaled)
                h.put(downscaled)
                h.put(downscaled)
                h.put(downscaled)
                climb_mask = cv2.inRange(hsv, lower_climb, upper_climb)
                contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 1 and keyboard.is_pressed('q'):
                    pyKey.press(key='r', sec=0.1)
                h.join()
                scale.join()
                flag.set()
                result.join()
                sys.stdout.write(f"\r{1 / (time.time() - start)}")
                sys.stdout.flush()

    def cargo(scale, result):
        lower_nocargo = np.array([0, 150, 144])
        upper_nocargo = np.array([1, 152, 146])
        lower_ycargo = np.array([25, 220, 120])
        upper_ycargo = np.array([35, 230, 130])
        lower_gcargo = np.array([70, 202, 152])
        upper_gcargo = np.array([72, 204, 154])
        while True:
            hsv = scale.get()
            ycargo_mask = cv2.inRange(hsv, lower_ycargo, upper_ycargo)
            nocargo_mask = cv2.inRange(hsv, lower_nocargo, upper_nocargo)
            gcargo_mask = cv2.inRange(hsv, lower_gcargo, upper_gcargo)
            if np.any(nocargo_mask):
                result.put(['c', 0])
            elif np.any(ycargo_mask):
                result.put(['c', 1])
            elif np.any(gcargo_mask):
                result.put(['c', 2])
            scale.task_done()
            result.join()

    def robot(h, result):
        lower_bumper_red = np.array([0, 169, 167])
        upper_bumper_red = np.array([40, 180, 222])
        lower_intake = np.array([0, 0, 0])
        upper_intake = np.array([60, 60, 60])
        lower_intake2 = np.array([20, 190, 120])
        upper_intake2 = np.array([55, 215, 160])
        while True:
            hsv = h.get()
            bumper_mask = cv2.inRange(hsv, lower_bumper_red, upper_bumper_red)
            bumper_mask = cv2.erode(bumper_mask, kernel, iterations=1)
            bumper_mask = cv2.dilate(bumper_mask, kernel, iterations=1)
            intake_mask = cv2.inRange(hsv, lower_intake, upper_intake)
            intake_mask2 = cv2.inRange(hsv, lower_intake2, upper_intake2)
            contours, _ = cv2.findContours(bumper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.concatenate(contours)
                rect = cv2.minAreaRect(all_points)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                M = cv2.moments(box)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                result.put(['c', (cX, cY)])

            intake_mask = intake_mask | intake_mask2
            intake_mask = intake_mask[0:540, 0:1280]
            contours, _ = cv2.findContours(intake_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.concatenate(contours)
                if len(all_points) > 0:
                    rect = cv2.minAreaRect(all_points)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    longest_side = max(np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4))

                    for i in range(4):
                        p1, p2 = box[i], box[(i + 1) % 4]
                        side_length = np.linalg.norm(p1 - p2)
                        if side_length == longest_side:
                            break

            if contours:
                intake_longest_side = max(cv2.arcLength(contour, True) for contour in contours)
                for contour in contours:
                    if cv2.arcLength(contour, True) == intake_longest_side:
                        rect = cv2.minAreaRect(contour)
                        angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                        angle_deg = math.degrees(angle_rad)
                        result.put(['a', angle_deg])
            h.task_done()
            result.join()

    def red_balls(h, result):
        lower_red = np.array([1, 202, 236])
        upper_red = np.array([3, 204, 238])
        while True:
            hsv = h.get()
            ball_mask = cv2.inRange(hsv, lower_red, upper_red)
            ball_mask = cv2.erode(ball_mask, kernel, iterations=1)
            ball_mask = cv2.dilate(ball_mask, kernel, iterations=1)
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pos = []
            if contours is not None:
                for cnt in contours:
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    if int(r) <= 20:
                        pos.append((int(x), int(y), int(r)))
                result.put(['rb', pos])
            h.task_done()
            result.join()

    def blue_balls(h, result):
        lower_blue = np.array([101, 244, 178])
        upper_blue = np.array([103, 246, 180])
        while True:
            hsv = h.get()
            ball_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            ball_mask = cv2.erode(ball_mask, kernel, iterations=1)
            ball_mask = cv2.dilate(ball_mask, kernel, iterations=1)
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pos = []
            if contours is not None:
                for cnt in contours:
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    if int(r) <= 20:
                        pos.append((int(x), int(y), int(r)))
                    result.put(['bb', pos])
            h.task_done()
            result.join()

    def detectScore(h, result):
        while True:
            hsv = h.get()
            frame = hsv[580:630, 440:690]
            score = ocr.ocr(frame, cls=True)
            try:
                score = ''.join(c for c in score[0][0][1][0] if c.isdigit())
                result.put(['s', score])
            except:
                pass
            h.task_done()
            result.join()


def start():
    i = 0
    try:
        vision_thread = multiprocessing.Process(target=vision.vision, args=(h, scale, result, flag))
        cargo_thread = multiprocessing.Process(target=vision.cargo, args=(scale, result))
        robot_thread = multiprocessing.Process(target=vision.robot, args=(h, result))
        red_balls_thread = multiprocessing.Process(target=vision.red_balls, args=(h, result))
        blue_balls_thread = multiprocessing.Process(target=vision.blue_balls, args=(h, result))
        score_thread = multiprocessing.Process(target=vision.detectScore, args=(h, result))
        vision_thread.start()
        cargo_thread.start()
        robot_thread.start()
        red_balls_thread.start()
        blue_balls_thread.start()
        score_thread.start()
        print(
            f"Vision alive: {vision_thread.is_alive()}, {vision_thread.pid} \nCargo alive: {cargo_thread.is_alive()}, {cargo_thread.pid} \nRobot alive: {robot_thread.is_alive()}, {robot_thread.pid} \nRed balls alive:  {red_balls_thread.is_alive()}, {red_balls_thread.pid} \nBlue balls alive: {blue_balls_thread.is_alive()}, {blue_balls_thread.pid} \nScore Alive: {score_thread.is_alive()}, {score_thread.pid}")
    except KeyboardInterrupt:
        print('Exiting...')
        vision_thread.terminate()
        cargo_thread.terminate()
        robot_thread.terminate()
        red_balls_thread.terminate()
        blue_balls_thread.terminate()
        score_thread.terminate()
        h.close()
        scale.close()
        result.close()
        cv2.destroyAllWindows()

def get_env(timestarted):
    start = None
    count = 0
    cargo = None
    center = None
    robot = None
    angle = None
    red_ball_pos = None
    blue_ball_pos = None
    score = None
    while not flag.is_set():
        pass
    qsize = result.qsize()
    while count < qsize:
        r = result.get()
        if r[0] == 'c':
            cargo = r[1]
        elif r[0] == 'r':
            center = r[1]
        elif r[0] == 'b':
            robot = r[1]
        elif r[0] == 'a':
            angle = r[1]
        elif r[0] == 'rb':
            red_ball_pos = r[1]
        elif r[0] == 'bb':
            blue_ball_pos = r[1]
        elif r[0] == 's':
            score = r[1]
        count += 1
    flag.clear()
    if timestarted == False and np.any(red_ball_pos):
        start = time.time()
        timestarted = True
    if timestarted == True and time.time() - start > 175:
        raise KeyboardInterrupt
    i = 0
    while i < qsize:
        result.task_done()
        i+=1

if __name__ == '__main__':
    timestarted = False
    start()
    while True:
        get_env(timestarted)
