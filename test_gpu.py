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
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CDQNagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model = CDQN(self.state_size, self.action_size).to(device)
        self.history_len = 4
        self.epsilon = 0.05  # 테스트에서도 작은 epsilon (같은 동작 무한 반복 방지)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(0, self.action_size)
        x = torch.as_tensor(state, device=device).float() / 255.0
        with torch.no_grad():
            return self.model(x).argmax(dim=1).item()

    def pre_processing(self, state):
        # obs_type="grayscale"이라 (210, 160) 흑백이 바로 들어옴 → 84x84로 리사이즈만
        return cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    gymnasium.register_envs(ale_py)
    env = gymnasium.make(
        "ALE/Breakout-v5",
        frameskip=4,  # 여기서 frame_skip 미리 때렸다.
        repeat_action_probability=0.0,
        obs_type="grayscale",  # 학습과 같은 전처리
        render_mode="human",
    )

    agent = CDQNagent(state_size=4, action_size=3)
    agent.model.load_state_dict(
        torch.load("./save_model/model_gpu.pt", map_location=device, weights_only=True)
    )
    agent.model.eval()

    EPISODE = 5
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
                print(f"episode {e} | score {score}")

    env.close()
