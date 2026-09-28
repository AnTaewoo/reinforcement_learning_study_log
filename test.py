from src import CDQN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gymnasium
import ale_py
from gymnasium.wrappers import RecordVideo
import pylab

from skimage.color import rgb2gray
from skimage.transform import resize


class CDQNagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model = CDQN(self.state_size, self.action_size)
        self.history_len = 4

    def get_action(self, state):
        state = torch.as_tensor(state / 255.0, dtype=torch.float32)
        x = torch.as_tensor(history / 255.0, dtype=torch.float32)
        with torch.no_grad():
            return self.model(x).argmax(dim=1).item()

    def pre_processing(self, state):
        return np.uint8(
            resize(rgb2gray(state), (84, 84), mode="constant") * 255
        )  # rgb를 하나의 gray_scale로 0~255의 값으로 처리.


if __name__ == "__main__":
    gymnasium.register_envs(ale_py)
    env = gymnasium.make(
        "ALE/Breakout-v5",
        frameskip=4,  # 여기서 frame_skip 미리 때렸다.
        repeat_action_probability=0.0,
        render_mode="human",
    )
    env.unwrapped.metadata["render_fps"] = 2000

    agent = CDQNagent(state_size=4, action_size=3)
    agent.model.load_state_dict(torch.load("./save_model/model.pt"))
    agent.model.eval()

    EPISODE = 10
    for e in range(EPISODE):
        state = env.reset()
        lives = state[1]["lives"]
        state = agent.pre_processing(state[0])
        history = np.stack((state, state, state, state))[np.newaxis]

        dead = True
        done = False
        score = 0
        while not done:
            action = agent.get_action(history)
            real_action = action + 1

            if dead:
                action = 0
                real_action = 1
                dead = False

            next_state, reward, terminated, truncated, info = env.step(real_action)
            done = terminated or truncated
            next_state = agent.pre_processing(next_state)
            next_history = np.append(
                next_state[np.newaxis][np.newaxis],
                history[:, :3, :, :],
                axis=1,
            )

            if lives > info["lives"]:
                dead = True
                lives = info["lives"]

            score += reward

            if dead:
                history = np.stack((next_state, next_state, next_state, next_state))[
                    np.newaxis
                ]
            else:
                history = next_history

            if done:
                print(
                    f"episode {e} | score {score} | memory_length {len(agent.replay_memory)} | epsilon {agent.epsilon}"
                )
