import pyautogui
import cv2
import numpy as np

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
            'balls_count': 0
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

            intake_mask = cv2.inRange(hsv, lower_intake, upper_intake)
            ycargo_mask = cv2.inRange(hsv, lower_ycargo, upper_ycargo)
            nocargo_mask = cv2.inRange(hsv, lower_nocargo, upper_nocargo)
            ball_mask = cv2.inRange(hsv, lower_red, upper_red)
            bumper_mask = cv2.inRange(hsv, lower_bumper_red, upper_bumper_red)
            kernel = np.ones((3, 3), np.uint8)
            bumper_mask = cv2.erode(bumper_mask, kernel, iterations=1)
            bumper_mask = cv2.dilate(bumper_mask, kernel, iterations=1)
            
            edges = cv2.Canny(ball_mask, 100, 200)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=25, minRadius=0, maxRadius=35)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                result['balls_count'] = len(circles)
                for (x, y, r) in circles:
                    result['balls_coords'].append((x, y))

            contours, _ = cv2.findContours(intake_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.concatenate(contours)
                if len(all_points) > 0:
                    rect = cv2.minAreaRect(all_points)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                    # Get the angle of the intake
                    angle = rect[2]
                    result['intake_angle'] = angle

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

            if np.any(nocargo_mask != 0):
                cv2.putText(frame, 'No cargo detected', org, font, fontScale, color, thickness, cv2.LINE_AA)
            elif np.any(ycargo_mask != 0):
                cv2.putText(frame, '1 cargo', org, font, fontScale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(frame, '2 cargo', org, font, fontScale, color, thickness, cv2.LINE_AA)

            yield frame
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()