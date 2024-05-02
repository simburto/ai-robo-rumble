import cv2
import numpy as np
import math
import threading
import pyKey
import time
from mss import mss
import multiprocessing
import keyboard

font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0)  
thickness = 2
kernel = np.ones((3, 3), np.uint8)
lock = threading.Lock()
exit_flag = threading.Event()
if __name__ == "__main__":
    result = multiprocessing.Manager().dict({
        'robot_center': None,
        'intake_angle': None,
        'red_ball_count': 0,
        'red_ball_pos': [],
        'climb_counter': None,
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

numbers = [
    cv2.imread('numbea/0.png'),
    cv2.imread('numbea/1.png'),
    cv2.imread('numbea/2.png'),
    cv2.imread('numbea/3.png'),
    cv2.imread('numbea/4.png'),
    cv2.imread('numbea/5.png'),
    cv2.imread('numbea/6.png'),
    cv2.imread('numbea/7.png'),
    cv2.imread('numbea/8.png'),
    cv2.imread('numbea/9.png'),
]

class vision():
    def vision(result):
        while True:
            climb_counter = 0
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            sct = mss()
            frame = np.array(sct.grab(monitor))
            result['frame'] = frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            result['hsv'] = hsv
            if np.any(hsv) and keyboard.is_pressed('q'):
                climb_counter = 0
                lower_climb = np.array([53,184,182]) 
                upper_climb = np.array([55,186,184])

                climb_mask = cv2.inRange(hsv, lower_climb, upper_climb)
                climb_mask = cv2.erode(climb_mask, kernel, iterations=1)
                climb_mask = cv2.dilate(climb_mask, kernel, iterations=1)

                contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    result['climb_box'] = contours
                    for cnt in contours:
                        climb_counter += 1
                    result['climb_counter'] = climb_counter
            cv2.waitKey(1)
            cv2.imshow('vision', frame)
    def cargo(result):
        while True:
            if np.any(result['hsv']):
                hsv = result['hsv']
                lower_nocargo = np.array([0,150,144])
                upper_nocargo = np.array([1,152,146])
                lower_ycargo = np.array([25,220,120])
                upper_ycargo = np.array([35,230,130])
                ycargo_mask = cv2.inRange(hsv, lower_ycargo, upper_ycargo)
                nocargo_mask = cv2.inRange(hsv, lower_nocargo, upper_nocargo)
                if np.any(nocargo_mask != 0):
                    result['cargo_count'] = 0
                elif np.any(ycargo_mask != 0):
                    result['cargo_count'] = 1
                else:
                    result['cargo_count'] = 2
    def robot(result):
        while True:
            if np.any(result['hsv']):
                hsv = result['hsv']
                cX = 0
                cY = 0
                lower_bumper_red = np.array([0, 169, 167])
                upper_bumper_red = np.array([37, 174, 220])
                bumper_mask = cv2.inRange(hsv, lower_bumper_red, upper_bumper_red)
                bumper_mask = cv2.erode(bumper_mask, kernel, iterations=1)
                bumper_mask = cv2.dilate(bumper_mask, kernel, iterations=1)
                lower_intake = np.array([0, 0, 0]) 
                upper_intake = np.array([5,5,5]) 
                intake_mask = cv2.inRange(hsv, lower_intake, upper_intake)
                
                contours, _ = cv2.findContours(bumper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_points = np.concatenate(contours)
                    rect = cv2.minAreaRect(all_points)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)  
                    result['bumper_box'] = box
                    M = cv2.moments(box)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    result['robot_center'] = (cX, cY)

                contours, _ = cv2.findContours(intake_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_points = np.concatenate(contours)
                    if len(all_points) > 0:
                        rect = cv2.minAreaRect(all_points)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        longest_side = max(np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4))
                        result['intake_box'] = box

                        for i in range(4):
                            p1, p2 = box[i], box[(i + 1) % 4]
                            side_length = np.linalg.norm(p1 - p2)
                            if side_length == longest_side:
                                result['intake_side'] = tuple(p1), tuple(p2)
                                break

                if contours:
                    intake_longest_side = max(cv2.arcLength(contour, True) for contour in contours)
                    for contour in contours:
                        if cv2.arcLength(contour, True) == intake_longest_side:
                            rect = cv2.minAreaRect(contour)
                            intake_box = cv2.boxPoints(rect)
                            intake_box = np.intp(intake_box)
                            angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                            angle_deg = math.degrees(angle_rad) 
                            result['intake_angle'] = angle_deg
    def red_balls(result):
        while True:
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
                    result['red_ball_box'] = circles
                    for (x, y, r) in circles:
                        result['red_ball_pos'].append((x, y))
    def blue_balls(result):
        while True:
            if np.any(result['hsv']):
                hsv = result['hsv']
                lower_blue = np.array([101, 244, 178])
                upper_blue = np.array([103, 246, 180])
                ball_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                edges = cv2.Canny(ball_mask,100,200)
                circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=0, maxRadius=35)
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    result['blue_ball_count'] = len(circles)
                    result['blue_ball_box'] = circles
                    for (x, y, r) in circles:
                        result['blue_ball_pos'].append((x, y))
    def display(result):
        while True:
            frame = result['frame']
            cargo_count = result['cargo_count']
            # Display cargo count
            if cargo_count == 0:
                text = 'No cargo detected'
            elif cargo_count == 1:
                text = '1 cargo'
            else:
                text = '2 cargo'
            cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

            if np.any(result['robot_center']) and np.any(result['intake_angle']):
                cX, cY = result['robot_center']
                ray_length = 100
                angle_rad = math.radians(result['intake_angle'])
                p2_x = int(cX + ray_length * math.cos(angle_rad))
                p2_y = int(cY + ray_length * math.sin(angle_rad))
                cv2.line(frame, (cX, cY), (p2_x, p2_y), (50, 255, 255), 2)

            if np.any(result['bumper_box']):
                for i in result['bumper_box']:
                    cv2.circle(frame, (i[0], i[1]), 10, (255, 255, 0), -1)
                cv2.drawContours(frame, [result['bumper_box']], 0, (255, 0, 0), 2)
                cv2.circle(frame, (cX, cY), 5, (36, 255, 12), -1)

            if np.any(result['intake_box']):
                cv2.drawContours(frame, [result['intake_box']], 0, (0, 255, 0), 2)

            if np.any(result['intake_side']):
                p1, p2 = result['intake_side']
                cv2.line(frame, p1, p2, (0, 0, 255), 2)
            if np.any(result['red_ball_box']):
                for (x, y, r) in result['red_ball_box']:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
            if np.any(result['blue_ball_box']):
                for (x, y, r) in result['blue_ball_box']:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
            if np.any(result['climb_box']):
                for cnt in result['climb_box']:
                            x,y,w,h = cv2.boundingRect(cnt)
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

            cv2.rectangle(frame, (0, 0), (500, 500), (50, 250, 250), 2)
            cv2.waitKey(1)
            cv2.imshow("Live", frame)

if __name__ == '__main__':
    timestarted = False
    i=0
    try:
        vision_thread = multiprocessing.Process(target=vision.vision, args=(result,))
        cargo_thread = multiprocessing.Process(target=vision.cargo, args=(result,))
        robot_thread = multiprocessing.Process(target=vision.robot, args=(result,))
        red_balls_thread = multiprocessing.Process(target=vision.red_balls, args=(result,))
        blue_balls_thread = multiprocessing.Process(target=vision.blue_balls, args=(result,))
        display_thread = multiprocessing.Process(target=vision.display, args=(result,))
        vision_thread.start()
        cargo_thread.start()
        robot_thread.start()
        red_balls_thread.start()
        blue_balls_thread.start()
        display_thread.start()
        while True:
            if result['climb_counter'] and result['climb_counter'] >1:
                pyKey.press(key='r',sec=0.1)
            if timestarted == False:
                start = time.time()
            elif result['red_ball_pos']:
                timestarted = True
            if time.time() - start > 155 and timestarted == True:
                pass
            i=i+1
            if i == 4000:
                print(f"Vision alive: {vision_thread.is_alive()} \nCargo alive: {cargo_thread.is_alive()} \nRobot alive: {robot_thread.is_alive()} \nRed balls alive:  {red_balls_thread.is_alive()} \nBlue balls alive: {blue_balls_thread.is_alive()} \nDisplay alive: {display_thread.is_alive()}")
                i=0

    except KeyboardInterrupt:
        print('Exiting...')
        vision_thread.terminate()
        cargo_thread.terminate()
        robot_thread.terminate()
        red_balls_thread.terminate()
        blue_balls_thread.terminate()
        display_thread.terminate()
        while display_thread.is_alive() == True:
            time.sleep(0.1)
        cv2.destroyAllWindows()