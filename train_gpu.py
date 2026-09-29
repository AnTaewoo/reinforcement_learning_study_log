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

# 아래 코드는 train.py의 학습 속도 저하 문제를 해결하기 위해 gpu를 사용하는 방법 입니다.


class CDQNagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model = CDQN(self.state_size, self.action_size).to(device)
        self.target_network = CDQN(self.state_size, self.action_size).to(device)
        self.learning_rate = 1e-4
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.replay_memory = deque(maxlen=100000)

        self.discount_v = 1.0 - 1e-2
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.epsilon_step_n = 1e6
        self.epsilon = self.epsilon_start
        self.epsilon_decay_step = (
            self.epsilon_start - self.epsilon_end
        ) / self.epsilon_step_n

        self.batch_size = 64
        self.history_len = 4
        self.skip_frame = 4

        self.train_start = 1e4
        self.train_freq = 4  # 4스텝마다 한 번 학습
        self.sync_target_network_rate = 1e4

        self.target_network.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(0, self.action_size)
        else:
            x = torch.as_tensor(state, device=device).float() / 255.0
            with torch.no_grad():
                return self.model(x).argmax(dim=1).item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.model.state_dict())

    def append_replay_memory(self, history, action, reward, next_history, deads):
        self.replay_memory.append((history, action, reward, next_history, deads))

    def train_model(self):
        # epsilon 감소는 메인 루프로 옮김 (train_freq만큼 느려지지 않도록)
        batch = random.sample(self.replay_memory, self.batch_size)

        # uint8 그대로 GPU로 보낸 뒤 GPU에서 float 변환 (전송량 1/4)
        history = (
            torch.as_tensor(
                np.array([sample[0][0] for sample in batch]), device=device
            ).float()
            / 255.0
        )
        next_history = (
            torch.as_tensor(
                np.array([sample[3][0] for sample in batch]), device=device
            ).float()
            / 255.0
        )
        actions = torch.as_tensor(
            np.array([sample[1] for sample in batch]), dtype=torch.int64, device=device
        )
        rewards = torch.as_tensor(
            np.array([sample[2] for sample in batch]),
            dtype=torch.float32,
            device=device,
        )
        deads = torch.as_tensor(
            np.array([sample[4] for sample in batch]),
            dtype=torch.float32,
            device=device,
        )  # model 들어가기 전 dtype 맞추기. (uint8로 하니 에러 발생)

        with torch.no_grad():
            targets = (
                rewards
                + (1 - deads)
                * self.discount_v
                * self.target_network(next_history).max(dim=1).values
            )

        q_values = self.model(history).gather(1, actions.unsqueeze(1)).squeeze(1)

        MSE = torch.square(targets - q_values).mean()  # batch 내 평균

        self.optim.zero_grad()
        MSE.backward()
        self.optim.step()

    def pre_processing(self, state):
        # obs_type="grayscale"이라 (210, 160) 흑백이 바로 들어옴 → 84x84로 리사이즈만
        return cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    print("device:", device)

    gymnasium.register_envs(ale_py)
    env = gymnasium.make(
        "ALE/Breakout-v5",
        frameskip=4,  # 여기서 frame_skip 미리 때렸다.
        repeat_action_probability=0.0,
        obs_type="grayscale",  # ALE가 흑백 화면을 직접 반환
        # render_mode="human",
    )
    # env.unwrapped.metadata["render_fps"] = 2000

    agent = CDQNagent(state_size=4, action_size=3)
    scores, episodes = [], []
    EPISODE = 15000
    step = 0
    score_avg = 0
    for e in range(EPISODE):
        state = env.reset()
        lives = state[1]["lives"]
        state = agent.pre_processing(state[0])
        history = np.stack((state, state, state, state))[np.newaxis]

        dead = True
        done = False
        score = 0
        while not done:
            step += 1

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
            )  # :3으로 끊으니 자동으로 index가 큰 과거 데이터는 삭제

            if lives > info["lives"]:
                dead = True
                lives = info["lives"]

            score += reward

            reward = (
                reward / 10.0 if not dead else -1
            )  # layer에 따른 block의 값을 dead = -1인 것에 비례하게 학습되기 위하여 나눠줌

            agent.append_replay_memory(history, action, reward, next_history, dead)

            if len(agent.replay_memory) > agent.train_start:
                # epsilon은 학습 주기와 상관없이 매 스텝 감소
                if agent.epsilon > agent.epsilon_end:
                    agent.epsilon -= agent.epsilon_decay_step

                if step % agent.train_freq == 0:
                    agent.train_model()

            if step % agent.sync_target_network_rate == 0:
                agent.update_target_network()

            if dead:
                history = np.stack((next_state, next_state, next_state, next_state))[
                    np.newaxis
                ]
            else:
                history = next_history

            if done:
                score_avg = 0.9 * score_avg + 0.1 * score

                scores.append(score_avg)
                episodes.append(e)
                print(
                    f"episode {e} | score {score} | memory_length {len(agent.replay_memory)} | epsilon {agent.epsilon}"
                )

                if e % 100 == 0:
                    pylab.clf()  # 이전 선을 지우고 다시 그림
                    pylab.plot(episodes, scores, "b")
                    pylab.xlabel("episode")
                    pylab.ylabel("score")
                    pylab.savefig("./save_graph/graph.png")

        if e % 1000 == 0:
            score_max = score
            torch.save(agent.model.state_dict(), "save_model/model_gpu.pt")
