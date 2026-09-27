import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pylab
import gymnasium as gym


class A2C(nn.Module):
    def __init__(self, state_size, action_size):
        super(A2C, self).__init__()
        self.actor_model = nn.Sequential(
            nn.Linear(state_size, 30),
            nn.ReLU(),
            nn.Linear(30, action_size),
            nn.Softmax(dim=-1),
        )
        self.critic_model = nn.Sequential(
            nn.Linear(state_size, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )

    def forward(self, x):
        policy_list = self.actor_model(x)
        value_list = self.critic_model(x)

        return policy_list, value_list


class A2Cagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model = A2C(state_size, action_size)

    def get_action(self, state):
        policy_list = self.model.actor_model(
            torch.as_tensor(state, dtype=torch.float32)
        )
        return torch.distributions.Categorical(probs=policy_list).sample().item()


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    env.unwrapped.metadata["render_fps"] = 2000
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2Cagent(state_size, action_size)
    agent.model.load_state_dict(torch.load("./save_model/model.pt"))
    agent.model.eval()
    scores, episodes = [], []

    EPISODE = 50
    for e in range(EPISODE):
        state = env.reset()
        state = torch.reshape(torch.as_tensor(state[0]), [1, state_size])
        score = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.reshape(torch.as_tensor(next_state), [1, state_size])

            done = terminated or truncated
            score += reward
            reward = 0.1 if not terminated else -1
            state = next_state
            if done:

                scores.append(score)
                episodes.append(e)

                print(f"episode {e} | score {score}")
