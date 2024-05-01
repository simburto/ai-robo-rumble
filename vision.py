import pyautogui
import cv2
import numpy as np
import math
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0)  
thickness = 2

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

def find_numbers_in_scoreframe(scoreframe):
    numbers_found = []

    for i, template in enumerate(numbers):
        res = cv2.matchTemplate(scoreframe, template, cv2.TM_SQDIFF)
        threshold = 0.5
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            numbers_found.append((i, pt))

    return numbers_found

def vision():
    cX = 0
    cY = 0
    while True:
        result = {
            'balls_coords': [],
            'robot_center': None,
            'intake_angle': None,
            'balls_count': 0,
            'climb_counter': None,
            'cargo_count': 0,
            'balls_shot': 0,
            'score': 0,
        }
        img = pyautogui.screenshot()

        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([1, 202, 236])
        upper_red = np.array([3, 204, 238])
        lower_bumper_red = np.array([0, 169, 167])
        upper_bumper_red = np.array([37, 174, 220])
        lower_nocargo = np.array([0,150,144])
        upper_nocargo = np.array([1,152,146])
        lower_ycargo = np.array([25,220,120])
        upper_ycargo = np.array([35,230,130])
        lower_intake = np.array([0, 0, 0]) 
        upper_intake = np.array([1,1,1]) 
        lower_climb = np.array([53,184,182]) 
        upper_climb = np.array([55,186,184])

        climb_mask = cv2.inRange(hsv, lower_climb, upper_climb)
        intake_mask = cv2.inRange(hsv, lower_intake, upper_intake)
        ycargo_mask = cv2.inRange(hsv, lower_ycargo, upper_ycargo)
        nocargo_mask = cv2.inRange(hsv, lower_nocargo, upper_nocargo)
        ball_mask = cv2.inRange(hsv, lower_red, upper_red)
        bumper_mask = cv2.inRange(hsv, lower_bumper_red, upper_bumper_red)
        kernel = np.ones((3, 3), np.uint8)
        bumper_mask = cv2.erode(bumper_mask, kernel, iterations=1)
        bumper_mask = cv2.dilate(bumper_mask, kernel, iterations=1)
        climb_mask = cv2.erode(climb_mask, kernel, iterations=1)
        climb_mask = cv2.dilate(climb_mask, kernel, iterations=1)

        scoreframe = frame[688:1080, 710:1210]
        cv2.rectangle(scoreframe,(0,0), (145, 402),  (0,0,0), -1)
        cv2.rectangle(scoreframe,(0,0), (550, 285),  (0,0,0), -1)
        cv2.rectangle(scoreframe,(321,0), (550, 402),  (0,0,0), -1)
        cv2.rectangle(scoreframe,(0,375), (550, 400),  (0,0,0), -1) 
        numbers_found = find_numbers_in_scoreframe(scoreframe)
   #     print("Extracted numbers from scoreframe:", numbers_found)

        cv2.imshow("haeiuthoa", scoreframe)

        edges = cv2.Canny(ball_mask,100,200)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=0, maxRadius=35)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            result['balls_count'] = len(circles)
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
                result['balls_coords'].append((x, y))

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
                box = np.int0(box)
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
                    intake_box = np.int0(intake_box)
                    angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    angle_deg = math.degrees(angle_rad) 
                    result['intake_angle'] = angle_deg
                    ray_length = 100 
                    p2_x = int(cX + ray_length * math.cos(angle_rad))
                    p2_y = int(cY + ray_length * math.sin(angle_rad))
                    cv2.line(frame, bumper_center, (p2_x, p2_y), (50, 255, 255), 2)

        contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            climb_counter = 0
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                climb_counter += 1
            result['climb_counter'] = climb_counter

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

        yield result
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()