import cv2
import numpy as np
import math
import pyKey
import time
import bettercam
import multiprocessing
import keyboard
import sys
import gymnasium as gym
from gymnasium import spaces

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
kernel = np.ones((3, 3), np.uint8)

h = multiprocessing.JoinableQueue()
scale = multiprocessing.JoinableQueue()
result = multiprocessing.JoinableQueue()
flag = multiprocessing.Event()


class Vision:
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
        lower_noCargo = np.array([0, 150, 144])
        upper_noCargo = np.array([1, 152, 146])
        lower_yCargo = np.array([25, 220, 120])
        upper_yCargo = np.array([35, 230, 130])
        lower_gCargo = np.array([70, 202, 152])
        upper_gCargo = np.array([72, 204, 154])
        while True:
            hsv = scale.get()
            yCargo_mask = cv2.inRange(hsv, lower_yCargo, upper_yCargo)
            noCargo_mask = cv2.inRange(hsv, lower_noCargo, upper_noCargo)
            gCargo_mask = cv2.inRange(hsv, lower_gCargo, upper_gCargo)
            if np.any(noCargo_mask):
                result.put(['c', 0])
            elif np.any(yCargo_mask):
                result.put(['c', 1])
            elif np.any(gCargo_mask):
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
                result.put(['r', (cX, cY)])

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


vision_thread = multiprocessing.Process(target=Vision.vision, args=(h, scale, result, flag))
cargo_thread = multiprocessing.Process(target=Vision.cargo, args=(scale, result))
robot_thread = multiprocessing.Process(target=Vision.robot, args=(h, result))
red_balls_thread = multiprocessing.Process(target=Vision.red_balls, args=(h, result))
blue_balls_thread = multiprocessing.Process(target=Vision.blue_balls, args=(h, result))
timeStarted = False
prevCargo = 0
count = 0


def _get_obs():
    start = None
    count = 0
    cargo = None
    center = None
    angle = None
    red_ball_pos = None
    blue_ball_pos = None
    while not flag.is_set():
        pass
    qsize = result.qsize()
    while count < qsize:
        r = result.get()
        if r[0] == 'c':
            cargo = r[1]
        elif r[0] == 'r':
            center = r[1]
        elif r[0] == 'a':
            angle = r[1]
        elif r[0] == 'rb':
            red_ball_pos = r[1]
        elif r[0] == 'bb':
            blue_ball_pos = r[1]
        count += 1
    flag.clear()
    global timeStarted
    if timeStarted is False and np.any(red_ball_pos):
        start = time.time()
        timeStarted = True
    i = 0
    while i < qsize:
        result.task_done()
        i += 1
    return {"cargo": cargo, "center": center, "angle": angle, "red_ball_pos": red_ball_pos,
            "blue_ball_pos": blue_ball_pos, "time": time.time() - start}


class RoboRumbleEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    timeStarted = False
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(7)
        vision_thread.start()
        cargo_thread.start()
        robot_thread.start()
        red_balls_thread.start()
        blue_balls_thread.start()
        self.observation_space = spaces.Dict(
            {
                "cargo": spaces.Discrete(3, start=0),
                "center": spaces.Box(low=np.array([0, 0]), high=np.array([1280, 720]), shape=(2,)),
                "angle": spaces.Box(low=-180, high=180, shape=(1,)),
                "red_ball_pos": spaces.Sequence(
                    spaces.Box(low=np.array([0, 0]), high=np.array([1280, 720]), shape=(2,))),
                "blue_ball_pos": spaces.Sequence(
                    spaces.Box(low=np.array([0, 0]), high=np.array([1280, 720]), shape=(2,))),
                "time": spaces.Box(low=0, high=180, shape=(1,)),
            }
        )

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        observation = _get_obs()
        cargo = observation['cargo']
        cargoDiff = prevCargo - cargo
        global prevCargo
        prevCargo = cargo
        if cargoDiff == 0:
            reward = -1
        else:
            reward = abs(cargoDiff)
        if observation['time'] > 175:
            terminated = True
        info = None
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        time.sleep(3)
        pyKey.press('SPACEBAR', sec=0.3)
        time.sleep(2)
        pyKey.press('r', sec=0.3)
        time.sleep(1)
        global timeStarted
        global prevCargo
        timeStarted = False
        prevCargo = 0
        observation = _get_obs()
        info = None
        return observation, info

    def close(self):
        vision_thread.terminate()
        cargo_thread.terminate()
        robot_thread.terminate()
        red_balls_thread.terminate()
        blue_balls_thread.terminate()
        h.close()
        scale.close()
        result.close()
