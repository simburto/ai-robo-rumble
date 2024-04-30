import pyautogui
import cv2
import numpy as np
import math

def vision():
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    fontScale = 1
    color = (255, 0, 0)  
    thickness = 2

    while True:
        result = {
            'balls_coords': [],
            'robot_center': None,
            'intake_angle': None,
            'balls_count': 0,
            'climb_counter': None,
            'endgame': False,
        }
        while True:
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
            lower_endgametimer = np.array([25,207,226])
            upper_endgametimer = np.array([27,209,228])

            endgametimer_mask = cv2.inRange(hsv, lower_endgametimer, upper_endgametimer)
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

            edges = cv2.Canny(ball_mask,100,200)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=0, maxRadius=35)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                result['balls_count'] = len(circles)
                for (x, y, r) in circles:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
                    result['balls_coords'].append((x, y))

            contours, _ = cv2.findContours(intake_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.concatenate(contours)
                if len(all_points) > 0:
                    # Get the minimum area rectangle enclosing the intake contour
                    rect = cv2.minAreaRect(all_points)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                    
                    # Get the longest side of the intake bounding box
                    longest_side = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
                    # Find the index of the corner points representing the longest side
                    longest_side_indices = np.where([np.linalg.norm(box[i] - box[(i+1)%4]) == longest_side for i in range(4)])[0]
                    # Calculate the angle of the longest side
                    if len(longest_side_indices) >= 2:
                        angle = math.atan2(box[longest_side_indices[1]][1] - box[longest_side_indices[0]][1], 
                                        box[longest_side_indices[1]][0] - box[longest_side_indices[0]][0]) * 180 / np.pi
                        result['intake_angle'] = angle
                    else:
                        result['intake_angle'] = None 

            contours, _ = cv2.findContours(bumper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.concatenate(contours)
                rect = cv2.minAreaRect(all_points)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
                # Get the center of the robot
                M = cv2.moments(all_points)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                result['robot_center'] = (cx, cy)

            contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                climb_counter = 0
                for cnt in contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    climb_counter = climb_counter+1
                result['climb_counter'] = climb_counter

            if np.any(nocargo_mask != 0):
                cv2.putText(frame, 'No cargo detected', org, font, fontScale, color, thickness, cv2.LINE_AA)
            elif np.any(ycargo_mask != 0):
                cv2.putText(frame, '1 cargo', org, font, fontScale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(frame, '2 cargo', org, font, fontScale, color, thickness, cv2.LINE_AA)

            if np.any(endgametimer_mask) != 0:
                result['endgame'] = True
            else:
                result['endgame'] = False

            result['frame'] = frame 

            yield result
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()