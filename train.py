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
            nn.Linear(30, 1),  # state value
        )

    def forward(self, x):
        policy_list = self.actor_model(x)
        value_list = self.critic_model(x)

        return policy_list, value_list


class A2Cagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_v = 0.99
        self.learning_rate = 0.001

        self.model = A2C(state_size, action_size)

       self.actor_optim = optim.Adam(
            self.model.actor_model.parameters(), lr=self.learning_rate
        )
        self.critic_optim = optim.Adam(
            self.model.critic_model.parameters(), lr=self.learning_rate
        ) 

    def get_action(self, state):
        with torch.no_grad():
            policy_list = self.model.actor_model(
                torch.as_tensor(state, dtype=torch.float32)
            )
        return (
            torch.distributions.Categorical(probs=policy_list).sample().item()
        )  # np.random.choice(self.action_size, 1, p=policy_list)는 float32특징 상 합이 1이 안될때 에러발생

    def train_model(self, state, action, reward, next_state, done, terminated):
        policy_list, value_list = self.model(state)
        _, next_value_list = self.model(next_state)
        policy = policy_list[0][action]

        # critic advantage
        with torch.no_grad():
            target = (
                reward + (1 - terminated) * self.discount_v * next_value_list
            )  # V(S_{t+1})이 V(S_t)로 내려가지 않기 위해.
        advantage = target - value_list
        MSEloss = torch.square(advantage)

        # actor
        CEloss = (
            -torch.log(policy) * advantage.detach()
        )  # actor-critic을 서로 다른 class로 정의했다면 detach 고려는 안해도 됐을 것 같다.

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        MSEloss.backward()
        CEloss.backward()
        self.actor_optim.step()
        self.critic_optim.step()

        return MSEloss, CEloss


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    env.unwrapped.metadata["render_fps"] = 2000
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2Cagent(state_size, action_size)
    scores, episodes = [], []
    score_avg = 0
    score_max = 0

    EPISODE = 750
    for e in range(EPISODE):
        state, _ = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, state_size)

        score = 0

        done = False
        while not done:
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
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
                pylab.savefig("./save_graph/graph.png")

        if e >= 100 and score_max < score:
            score_max = score
            torch.save(agent.model.state_dict(), "./save_model/model.pt")
