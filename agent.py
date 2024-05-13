# Import necessary libraries
import gym
import numpy as np
from stable_baselines3 import PPO

# Initialize the environment
env = gym.make("your_env_name")

# Define the observation space
observation_space = env.observation_space

# Define the action space
action_space = env.action_space

# Initialize the agent
model = PPO(policy="MlpPolicy", env=env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the agent
model.save("ppo_model")

# Load the agent
loaded_model = PPO.load("ppo_model")

# Test the agent
for i in range(10):
    state, done = env.reset(), False
    while not done:
        action = loaded_model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()