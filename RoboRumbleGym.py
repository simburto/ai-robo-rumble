import cv2
import numpy as np
import math
import time
import bettercam
import multiprocessing
import pyautogui
import gymnasium as gym
import sys
import keyboard

KERNEL = np.ones((3, 3), np.uint8)


class VisionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.h = multiprocessing.JoinableQueue()
        self.scale = multiprocessing.JoinableQueue()
        self.result = multiprocessing.JoinableQueue()
        self.flag = multiprocessing.Event()
        self.count = 0
        self.prev_pos = None
        self.action_space = gym.spaces.Discrete(3)
        self.ENV = None
        self.prev_cargo = None
        self.time_started = False
        self.start = None
        self.climb_reward_given = False
        self.prev_action = None
        self.started = False
        self.action_dict = ['w', 'a', 's', 'd', 'f', 'g', 'q', (['w', 'a']), (['w', 'd']), (['s', 'a']), (['s', 'd']),
                            (['w', 'f']), (['w', 'g']), (['s', 'f']), (['s', 'g']), (['d', 'f']), (['d', 'g']),
                            (['a', 'f']), (['a', 'g']), (['w', 'a', 'f']), (['w', 'a', 'g']), (['w', 'd', 'f']),
                            (['w', 'd', 'g']), (['s', 'a', 'f']), (['s', 'a', 'g']), (['s', 'd', 'f']),
                            (['s', 'd', 'g']),
                            ]
        self.action_space = gym.spaces.Discrete(27)
        self.observation_space = gym.spaces.Dict({
            "cargo": gym.spaces.Discrete(3),
            "center": gym.spaces.Box(low=0, high=1280, shape=(2,)),
            "angle": gym.spaces.Box(low=-180, high=180, shape=(1,)),
            "red_ball_pos": gym.spaces.Box(low=0, high=1280, shape=(1, 3)),
            "blue_ball_pos": gym.spaces.Box(low=0, high=1280, shape=(1, 3)),
            "time": gym.spaces.Box(low=0, high=200, shape=(1,))
        })

    def reset(self, seed=None, options=None):
        self.flag.clear()
        self.count = 0
        self.prev_pos = None
        pyautogui.press('space')
        time.sleep(2)
        pyautogui.press('r')
        if self.started is False:
            self.run()
            self.started = True
        observation = self._get_obs()
        return observation

    def step(self, action):
        if self.prev_action:
            pyautogui.keyUp(self.prev_action)
        pyautogui.keyDown(self.action_dict[action])
        self.prev_action = self.action_dict[action]
        terminated = False
        observation = self.ENV._get_obs()
        reward = 0
        info = None
        cargo = observation['cargo']
        shoot_count = observation['shoot_count']
        self.time_started = observation['time_started']
        if observation['climbed']:
            reward = 15
            self.climb_reward_given = True
        if self.time_started is True and not observation['climbed']:
            if shoot_count:
                reward = shoot_count ** (shoot_count - 1)
            if self.prev_cargo and cargo:
                cargo_diff = self.prev_cargo - cargo
                if cargo_diff == 0:
                    reward = reward - 0.01
                else:
                    reward = reward + abs(cargo_diff) * 10
            self.prev_cargo = cargo
            if self.prev_pos and observation['center']:
                movement = math.dist(self.prev_pos, observation['center'])
                if movement == 0:
                    reward = reward - 5
                else:
                    reward = reward + abs(movement) * 0.01
            self.prev_pos = observation['center']
            if observation['time'] > 155:
                terminated = True
        return observation, reward, terminated, False, info

    def _get_obs(self):
        count = 0
        cargo = None
        center = None
        angle = None
        red_ball_pos = None
        blue_ball_pos = None
        shoot_count = 0
        climbed = False
        while not self.flag.is_set():
            pass
        qsize = self.result.qsize()
        while count < qsize:
            r = self.result.get()
            if r[0] == "c":
                cargo = r[1]
            elif r[0] == "r":
                center = r[1]
            elif r[0] == "a":
                angle = r[1]
            elif r[0] == "rb":
                red_ball_pos = r[1]
            elif r[0] == "bb":
                blue_ball_pos = r[1]
            elif r[0] == "s":
                shoot_count = r[1]
            elif r[0] == "cl":
                climbed = r[1]
            count += 1
        self.flag.clear()
        if self.time_started is False:
            self.start = time.time()
            if red_ball_pos:
                self.time_started = True
        i = 0
        while i < count:
            self.result.task_done()
            i += 1
        return {
            "climbed": climbed,
            "time_started": self.time_started,
            "cargo": cargo,
            "center": center,
            "angle": angle,
            "red_ball_pos": red_ball_pos,
            "blue_ball_pos": blue_ball_pos,
            "time": time.time() - self.start,
            "shoot_count": shoot_count,
        }

    def run(self):
        self.ENV = VisionEnv()

        vision_thread = multiprocessing.Process(target=self.ENV.vision)
        cargo_thread = multiprocessing.Process(target=self.ENV.cargo)
        robot_thread = multiprocessing.Process(target=self.ENV.robot)
        red_balls_thread = multiprocessing.Process(target=self.ENV.red_balls)
        blue_balls_thread = multiprocessing.Process(target=self.ENV.blue_balls)
        print(__name__)
        vision_thread.start()
        cargo_thread.start()
        robot_thread.start()
        red_balls_thread.start()
        blue_balls_thread.start()

        print(
            f'Vision alive: {vision_thread.is_alive()}, {vision_thread.pid} \nCargo alive: {cargo_thread.is_alive()}, {cargo_thread.pid} \nRobot alive: {robot_thread.is_alive()}, {robot_thread.pid} \nRed balls alive:  {red_balls_thread.is_alive()}, {red_balls_thread.pid} \nBlue balls alive: {blue_balls_thread.is_alive()}, {blue_balls_thread.pid}')

    def vision(self):
        lower_climb = np.array([53, 184, 182])
        upper_climb = np.array([55, 186, 184])
        camera = bettercam.create(output_color="BGR")
        climbed = False
        while True:
            frame = np.array(camera.grab())
            if np.any(frame):
                start = time.time()
                frame = frame.astype(np.uint8)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                downscaled = cv2.resize(hsv, (1152, 648), interpolation=cv2.INTER_LINEAR)
                self.scale.put(hsv)
                self.h.put(downscaled)
                self.h.put(downscaled)
                self.h.put(downscaled)
                climb_mask = cv2.inRange(hsv, lower_climb, upper_climb)
                contours, _ = cv2.findContours(climb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 1 and keyboard.is_pressed('q'):
                    pyautogui.press('r')
                    climbed = True
                if climbed:
                    self.result.put(["cl", True])
                self.h.join()
                self.scale.join()
                self.flag.set()
                self.result.join()
                sys.stdout.write(f"\r{1 / (time.time() - start)}")
                sys.stdout.flush()

    def cargo(self):
        lower_nocargo = np.array([0, 150, 144])
        upper_nocargo = np.array([1, 152, 146])
        lower_ycargo = np.array([25, 220, 120])
        upper_ycargo = np.array([35, 230, 130])
        lower_gcargo = np.array([70, 202, 152])
        upper_gcargo = np.array([72, 204, 154])

        while True:
            hsv = self.scale.get()
            ycargo_mask = cv2.inRange(hsv, lower_ycargo, upper_ycargo)
            nocargo_mask = cv2.inRange(hsv, lower_nocargo, upper_nocargo)
            gcargo_mask = cv2.inRange(hsv, lower_gcargo, upper_gcargo)
            if np.any(nocargo_mask):
                self.result.put(["c", 0])
            elif np.any(ycargo_mask):
                self.result.put(["c", 1])
            elif np.any(gcargo_mask):
                self.result.put(["c", 2])
            self.scale.task_done()
            self.result.join()

    def robot(self):
        lower_bumper_red = np.array([0, 169, 167])
        upper_bumper_red = np.array([40, 180, 222])
        lower_intake = np.array([0, 0, 0])
        upper_intake = np.array([60, 60, 60])
        lower_intake2 = np.array([20, 190, 120])
        upper_intake2 = np.array([55, 215, 160])
        lower_red = np.array([1, 202, 236])
        upper_red = np.array([3, 204, 238])
        while True:
            hsv = self.h.get()
            bumper_mask = cv2.inRange(hsv, lower_bumper_red, upper_bumper_red)
            bumper_mask = cv2.erode(bumper_mask, KERNEL, iterations=1)
            bumper_mask = cv2.dilate(bumper_mask, KERNEL, iterations=1)
            intake_mask = cv2.inRange(hsv, lower_intake, upper_intake)
            intake_mask2 = cv2.inRange(hsv, lower_intake2, upper_intake2)
            contours, _ = cv2.findContours(bumper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.concatenate(contours)
                rect = cv2.minAreaRect(all_points)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                x, y, w, h = cv2.boundingRect(box)
                shooting_frame = hsv[y:y + h, x:x + w]
                if np.any(shooting_frame):
                    shooting_mask = cv2.inRange(shooting_frame, lower_red, upper_red)
                    shoot_contours, _ = cv2.findContours(shooting_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    self.result.put(["s", len(shoot_contours)])
                M = cv2.moments(box)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.result.put(["r", (cX, cY)])

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

                    intake_longest_side = max(cv2.arcLength(contour, True) for contour in contours)
                    for contour in contours:
                        if cv2.arcLength(contour, True) == intake_longest_side:
                            angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                            angle_deg = math.degrees(angle_rad)
                            self.result.put(["a", angle_deg])
            self.h.task_done()
            self.result.join()

    def red_balls(self):
        lower_red = np.array([1, 202, 236])
        upper_red = np.array([3, 204, 238])
        while True:
            hsv = self.h.get()
            ball_mask = cv2.inRange(hsv, lower_red, upper_red)
            ball_mask = cv2.erode(ball_mask, KERNEL, iterations=1)
            ball_mask = cv2.dilate(ball_mask, KERNEL, iterations=1)
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pos = []
            if contours is not None:
                for cnt in contours:
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    if int(r) <= 20:
                        pos.append((int(x), int(y), int(r)))
                self.result.put(["rb", pos])
            self.h.task_done()
            self.result.join()

    def blue_balls(self):
        lower_blue = np.array([101, 244, 178])
        upper_blue = np.array([103, 246, 180])

        while True:
            hsv = self.h.get()
            ball_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            ball_mask = cv2.erode(ball_mask, KERNEL, iterations=1)
            ball_mask = cv2.dilate(ball_mask, KERNEL, iterations=1)
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pos = []
            if contours is not None:
                for cnt in contours:
                    (x, y), r = cv2.minEnclosingCircle(cnt)
                    if int(r) <= 20:
                        pos.append((int(x), int(y), int(r)))
                    self.result.put(["bb", pos])
            self.h.task_done()
            self.result.join()

