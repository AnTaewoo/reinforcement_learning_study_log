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
        sigma = self.sigma_model(x) + 1e-3  # sigma > 0 조건
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
    def __init__(self, state_size, action_size, max_action):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_v = 0.99
        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3

        self.actor_model = CA(state_size, action_size)
        self.critic_model = CC(state_size)

        self.max_action = max_action

        self.actor_optim = optim.Adam(
            self.actor_model.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optim = optim.Adam(
            self.critic_model.parameters(), lr=self.critic_learning_rate
        )

    def get_action(self, state):
        mu, sigma = self.actor_model(state)
        dist = Normal(loc=mu, scale=sigma)
        act = dist.sample()
        act = np.clip(act, -self.max_action, self.max_action)
        return act

    def train_model(self, state, action, reward, next_state, done, terminated):
        mu, sigma = self.actor_model(state)

        log_policy = Normal(loc=mu, scale=sigma).log_prob(action).sum(dim=-1)
        value = self.critic_model(state)
        next_value = self.critic_model(next_state)

        with torch.no_grad():
            target = reward + (1 - terminated) * self.discount_v * next_value

        advantage = target - value
        MSEloss = torch.square(advantage)

        CEloss = (
            -log_policy * advantage.detach()
        )  # -log(act)의 경우 dist를 train_model에서 호출 시, dist.log_prob(act)를 통해 쉽게 구현할 수 있다
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        MSEloss.backward()
        CEloss.backward()
        self.actor_optim.step()
        self.critic_optim.step()

        return MSEloss, CEloss


if __name__ == "__main__":
    gym.envs.register(
        id="CartPoleContinuous-v0",
        entry_point="env:ContinuousCartPoleEnv",
        max_episode_steps=500,
        reward_threshold=480,
    )
    env = gym.make("CartPoleContinuous-v0", render_fps=4000)

    state_size = env.observation_space.shape[0]
    action_size = 1

    agent = CA2Cagent(state_size, action_size=1, max_action=env.action_space.high[0])

    scores, episodes = [], []
    score_avg = 0
    score_max = 0

    EPISODE = 2500
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
            reward = 0.1 if not terminated else -1

            MSEloss, CEloss = agent.train_model(
                state, action, reward, next_state, done, terminated
            )

            state = next_state

            if done:
                score_avg = 0.9 * score_avg + 0.1 * score

                scores.append(score_avg)
                episodes.append(e)

                print(
                    f"episode {e} | score {score} | actor loss {CEloss[0][0]:.2f} | critic loss {MSEloss[0][0]:.2f}"
                )

                pylab.plot(episodes, scores, "b")
                pylab.xlabel("episode")
                pylab.ylabel("score")
                # pylab.savefig("./save_graph/continuous_a2c_graph.png")

        if e >= 100 and score_max <= score and e % 100 == 0:
            score_max = score
            torch.save(
                agent.actor_model.state_dict(),
                "./save_model/continuous_actor_model.pt",
            )
            torch.save(
                agent.critic_model.state_dict(),
                "./save_model/continuous_critic_model.pt",
            )
