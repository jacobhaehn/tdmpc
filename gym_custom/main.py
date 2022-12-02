import gym
from gym.wrappers import AtariPreprocessing as AtP
import cv2 as cv
import numpy
# import iris
# from PIL import Image
def policy():
    print()

task = "Pong-v0"

env = AtP(gym.make(task, render_mode="rgb_array"), frame_skip=1)

# obs, info = env.reset(seed=42)

obs = env.reset(seed=42)
print(obs.shape)

# cap = cv.VideoCapture(0)

for _ in range(1000):
    obs, reward, terminated, truncated = env.step(env.action_space.sample())
    # img = Image.new("L", obs.shape)
    # img.show()
    # butt = env.render(mode='rgb_array')
    if terminated or truncated:
        obs = env.reset(seed=42)

env.close()





# print(info)