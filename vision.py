import pyautogui
import cv2
import numpy as np
import math
import threading
import pyKey
import time

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", 1280,720)

font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0)  
thickness = 2
kernel = np.ones((3, 3), np.uint8)

lock = threading.Lock()
exit_flag = threading.Event()

global result
result = {
        'robot_center': None,
        'intake_angle': None,
        'red_ball_count': 0,
        'red_ball_pos': [],
        'climb_counter': None,
        'cargo_count': 0,
        'blue_ball_count': 0,
        'blue_ball_pos': [],
        'frame': None,
        'hsv': None,
        }

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
    def autoclimb(needlebboxcount):
        if needlebboxcount > 1:
            pyKey.press(key='r',sec=0.1)
    def start():
        global cargo_thread 
        global robot_thread 
        global red_balls_thread 
        global climb_thread 
        global blue_balls_thread 
        global vision_thread
        vision_thread = threading.Thread(target=vision.vision, args=(result, lock, exit_flag))
        vision_thread.start()
        time.sleep(1)
        cargo_thread = threading.Thread(target=vision.cargo, args=(result, lock, exit_flag))
        robot_thread = threading.Thread(target=vision.robot, args=(result, lock, exit_flag))
        red_balls_thread = threading.Thread(target=vision.red_balls, args=(result, lock, exit_flag))
        climb_thread = threading.Thread(target=vision.climb, args=(result, lock, exit_flag))
        blue_balls_thread = threading.Thread(target=vision.blue_balls, args=(result, lock, exit_flag))
    
        cargo_thread.start()
        robot_thread.start()
        red_balls_thread.start()
        climb_thread.start()
        blue_balls_thread.start()
    def vision(result, lock, exit_flag):
        status = False
        while not exit_flag.is_set():
            with lock:
                if status == False:
                    print("Vision thread started.")
                    status = True
                img = pyautogui.screenshot()
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                result['hsv'] = hsv
                result['frame'] = frame 
    def cargo(result, lock, exit_flag):
        status = False
        while not exit_flag.is_set() and np.any(result['hsv']) and np.any(result['frame']):
            with lock:
                if status == False:
                    print("Cargo thread started.")
                    status = True
                frame = result['frame']
                hsv = result['hsv']
                lower_nocargo = np.array([0,150,144])
                upper_nocargo = np.array([1,152,146])
                lower_ycargo = np.array([25,220,120])
                upper_ycargo = np.array([35,230,130])
                ycargo_mask = cv2.inRange(hsv, lower_ycargo, upper_ycargo)
                nocargo_mask = cv2.inRange(hsv, lower_nocargo, upper_nocargo)
                if np.any(nocargo_mask != 0):
                        cv2.putText(frame, 'No cargo detected', org, font, fontScale, color, thickness, cv2.LINE_AA)
                        result['cargo_count'] = 0
                elif np.any(ycargo_mask != 0):
                    cv2.putText(frame, '1 cargo', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    result['cargo_count'] = 1
                else:
                    cv2.putText(frame, '2 cargo', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    result['cargo_count'] = 2
                result['frame'] = frame 
    def robot(result, lock, exit_flag):
        status = False
        while not exit_flag.is_set() and np.any(result['hsv']) and np.any(result['frame']):
            with lock:
                if status == False:
                    print("Robot thread started.")
                    status = True
                frame = result['frame']
                hsv = result['hsv']
                cX = 0
                cY = 0
                lower_bumper_red = np.array([0, 169, 167])
                upper_bumper_red = np.array([37, 174, 220])
                bumper_mask = cv2.inRange(hsv, lower_bumper_red, upper_bumper_red)
                bumper_mask = cv2.erode(bumper_mask, kernel, iterations=1)
                bumper_mask = cv2.dilate(bumper_mask, kernel, iterations=1)
                lower_intake = np.array([0, 0, 0]) 
                upper_intake = np.array([1,1,1]) 
                intake_mask = cv2.inRange(hsv, lower_intake, upper_intake)
                
                contours, _ = cv2.findContours(bumper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_points = np.concatenate(contours)
                    rect = cv2.minAreaRect(all_points)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)  
                    for i in box:
                        cv2.circle(frame,(i[0],i[1]), 10, (255,255,0), -1)
                    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
                    M = cv2.moments(box)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cX, cY), 5, (36, 255, 12), -1)
                    result['robot_center'] = (cX, cY)

                contours, _ = cv2.findContours(intake_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_points = np.concatenate(contours)
                    if len(all_points) > 0:
                        rect = cv2.minAreaRect(all_points)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        longest_side = max(np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4))
                        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                        for i in range(4):
                            p1, p2 = box[i], box[(i + 1) % 4]
                            side_length = np.linalg.norm(p1 - p2)
                            if side_length == longest_side:
                                cv2.line(frame, tuple(p1), tuple(p2), (0, 0, 255), 2)
                                break

                if contours:
                    bumper_center = (cX, cY)

                    intake_longest_side = max(cv2.arcLength(contour, True) for contour in contours)
                    for contour in contours:
                        if cv2.arcLength(contour, True) == intake_longest_side:
                            rect = cv2.minAreaRect(contour)
                            intake_box = cv2.boxPoints(rect)
                            intake_box = np.intp(intake_box)
                            angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                            angle_deg = math.degrees(angle_rad) 
                            result['intake_angle'] = angle_deg
                            ray_length = 100 
                            p2_x = int(cX + ray_length * math.cos(angle_rad))
                            p2_y = int(cY + ray_length * math.sin(angle_rad))
                            cv2.line(frame, bumper_center, (p2_x, p2_y), (50, 255, 255), 2)
                result['frame'] = frame 
    def red_balls(result, lock, exit_flag):
        status = False
        while not exit_flag.is_set() and np.any(result['hsv']) and np.any(result['frame']):
            with lock:
                if status == False:
                    print("Red balls thread started.")
                    status = True
                frame = result['frame']
                hsv = result['hsv']
                lower_red = np.array([1, 202, 236])
                upper_red = np.array([3, 204, 238])
                ball_mask = cv2.inRange(hsv, lower_red, upper_red)
                edges = cv2.Canny(ball_mask,100,200)
                circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=0, maxRadius=35)
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    result['red_ball_count'] = len(circles)
                    for (x, y, r) in circles:
                        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                        cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
                        result['red_ball_pos'].append((x, y))
                result['frame'] = frame 
    def climb(result, lock, exit_flag):
        status = False
        while not exit_flag.is_set() and np.any(result['hsv']) and np.any(result['frame']):
            with lock:
                if status == False:
                    print("Climb thread started.")
                    status = True
                frame = result['frame']
                hsv = result['hsv']
                lower_climb = np.array([53,184,182]) 
                upper_climb = np.array([55,186,184])

                climb_mask = cv2.inRange(hsv, lower_climb, upper_climb)
                climb_mask = cv2.erode(climb_mask, kernel, iterations=1)
                climb_mask = cv2.dilate(climb_mask, kernel, iterations=1)

                contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    climb_counter = 0
                    for cnt in contours:
                        x,y,w,h = cv2.boundingRect(cnt)
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                        climb_counter += 1
                    result['climb_counter'] = climb_counter
                result['frame'] = frame 
    def blue_balls(result, lock, exit_flag):
        status = False
        while not exit_flag.is_set() and np.any(result['hsv']) and np.any(result['frame']):
            with lock:
                if status == False:
                    print("Blue balls thread started.")
                    status = True
                frame = result['frame']
                hsv = result['hsv']
                lower_blue = np.array([101, 244, 178]) # change colours
                upper_blue = np.array([103, 246, 180])
                ball_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                edges = cv2.Canny(ball_mask,100,200)
                circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=0, maxRadius=35)
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    result['blue_ball_count'] = len(circles)
                    for (x, y, r) in circles:
                        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                        cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
                        result['blue_ball_pos'].append((x, y))
                result['frame'] = frame 

def main():
    timestarted = False
    try:
        vision.start()
        while True:
            print(result)
            if result['climb_counter']:
                vision.autoclimb(result['climb_counter'])
            if np.any(result['frame']):
                cv2.imshow("Live", result['frame'])
            if timestarted == False:
                start = time.time()
            if result['red_ball_pos']:
                timestarted = True
            cv2.rectangle(frame, (0,0), (500,500), (50,250,250), 2)
            if time.time() - start > 155 and timestarted == True:
                pass
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        exit_flag.set()
        cargo_thread.join()
        robot_thread.join()
        red_balls_thread.join()
        climb_thread.join()
        blue_balls_thread.join()

main()