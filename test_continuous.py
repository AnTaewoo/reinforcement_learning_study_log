import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pylab
import gymnasium as gym
from torch.distributions import Normal


class CA(nn.Module):
    def __init__(self, state_size, action_size):
        super(CA, self).__init__()
        self.actor_model_base = nn.Sequential(
            nn.Linear(state_size, 30),
            nn.ReLU(),
        )
        self.mu_model = nn.Sequential(
            nn.Linear(30, action_size),
        )
        self.sigma_model = nn.Sequential(
            nn.Linear(30, action_size),
            nn.Sigmoid(),  # 표준편차이므로 0~1의 값을 가지도록.
        )

    def forward(self, x):
        x = self.actor_model_base(x)
        mu = self.mu_model(x)
        sigma = self.sigma_model(x) + 1e-2  # sigma > 0 조건
        return mu, sigma


class CC(nn.Module):
    def __init__(self, state_size):
        super(CC, self).__init__()
        self.critic_model = nn.Sequential(
            nn.Linear(state_size, 30),
            nn.ReLU(),
            nn.Linear(30, 1),  # state value
        )

    def forward(self, x):
        return self.critic_model(x)


class CA2Cagent:
    def __init__(self, state_size, action_size, max_action=1):
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action

        self.actor_model = CA(state_size, action_size)

        self.actor_model.load_state_dict(
            torch.load("./save_model/continuous_actor_model.pt")
        )  # action에 대한 값만 얻으면 되기 때문에, 정책 업데이트 식을 완성할 필요 없으므로 critic_model은 안불러와도 된다~

    def get_action(self, state):
        with torch.no_grad():
            mu, sigma = self.actor_model(state)
        mu = np.clip(mu, -self.max_action, self.max_action)
        return mu


if __name__ == "__main__":
    gym.envs.register(
        id="CartPoleContinuous-v0",
        entry_point="env:ContinuousCartPoleEnv",
        max_episode_steps=1000,
        reward_threshold=980,
    )
    env = gym.make("CartPoleContinuous-v0", render_mode="human", render_fps=200)

    state_size = env.observation_space.shape[0]
    action_size = 1

    agent = CA2Cagent(state_size, action_size=1, max_action=env.action_space.high[0])

    scores, episodes = [], []
    score_avg = 0

    EPISODE = 10
    for e in range(EPISODE):
        state, _ = env.reset()
        state = torch.reshape(torch.as_tensor(state), [1, state_size])

        score = 0

        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(np.array(action[0]))
            next_state = torch.reshape(torch.as_tensor(next_state), [1, state_size])

            done = terminated or truncated
            score += reward

            state = next_state

            if done:
                score_avg = 0.9 * score_avg + 0.1 * score

                scores.append(score_avg)
                episodes.append(e)

                print(f"episode {e} | score {score}")
