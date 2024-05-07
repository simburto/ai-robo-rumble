import bettercam
import cv2
import numpy as np
camera = bettercam.create(output_color="BGR")
kernel = np.ones((3, 3), np.uint8)
while True:
    frame = np.array(camera.grab())
    if np.any(frame):
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (1152, 648), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_intake = np.array([0, 0, 0])
        upper_intake = np.array([50, 50, 65])
        lower_nocargo = np.array([53,210,156])
        upper_nocargo = np.array([55, 212, 158])
        ycargo_mask = cv2.inRange(hsv, lower_intake, upper_intake)
        nocargo_mask = cv2.inRange(hsv, lower_nocargo, upper_nocargo)
        mask = ycargo_mask|nocargo_mask
        mask = cv2.dilate(mask, kernel, iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None:
            for cnt in contours:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 4)
        cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)