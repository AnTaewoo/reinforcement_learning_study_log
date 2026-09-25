import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class REINFORCE(tf.keras.Model):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = Dense(30, activation="relu")
        self.fc2 = Dense(30, activation="relu")
        self.fc3 = Dense(action_size, activation="softmax")

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class REINFORCEagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.9
        self.learning_rate = 0.001
        self.model = REINFORCE(self.action_size)
        self.optimizers = Adam(lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policy = np.array(self.model(state)[0])
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discount_rewards = np.zeros_like(rewards)

        reward = 0
        for i, r in enumerate(reversed(rewards)):
            reward = r + self.discount_factor * reward
            discount_rewards[i] = reward
        return np.array(list(reversed(discount_rewards)))

    def append_sample(self, state, action, reward):
        self.states.append(state[0])  # 배치때문에 [0]을 해줘야 한다;;;
        self.rewards.append(reward)
        act = np.zeros(self.action_size, dtype=np.float32)
        act[action] = 1
        self.actions.append(act)

    def train_model(self):
        _discount_rewards = np.float32(self.discount_rewards(self.rewards))
        discount_rewards = (_discount_rewards - np.mean(_discount_rewards)) / np.std(
            _discount_rewards
        )

        model_param = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_param)

            policies = self.model(np.array(self.states))
            actions = self.actions

            action_prob = tf.reduce_sum(actions * policies, axis=1)
            log_policies = -tf.math.log(
                action_prob + 1e-8
            )  # cross-entropy랑 동일, 1e-8을 통해 policy NaN 방지
            loss = tf.reduce_sum(log_policies * discount_rewards)

        grads = tape.gradient(loss, model_param)
        self.optimizers.apply_gradients(zip(grads, model_param))
        self.states, self.actions, self.rewards = [], [], []

        return np.mean(log_policies)


def array_parse_to_batch(arr):
    return np.reshape(arr, [1, len(arr)])


if __name__ == "__main__":
    env = Env(render_speed=0.01)
    state_size = 15
    action_list = [0, 1, 2, 3, 4]
    action_size = len(action_list)
    agent = REINFORCEagent(state_size, action_size)

    EPISODES = 200
    scores, episodes = [], []
    for e in range(EPISODES):
        state = array_parse_to_batch(env.reset())
        done = False
        score = 0

        state = env.reset()
        state = array_parse_to_batch(state)  # batch 관리

        done = False
        while not done:
            action = np.array(agent.get_action(state))
            next_state, reward, done = env.step(action)
            next_state = array_parse_to_batch(next_state)

            agent.append_sample(next_state, action, reward)
            score += reward

            state = next_state

        entropy = agent.train_model()
        print(f"episode: {e} | score: {score} | entropy: {entropy}")

        scores.append(score)
        episodes.append(e)
        pylab.plot(episodes, scores, "b")
        pylab.xlabel("episode")
        pylab.ylabel("score")
        pylab.savefig("./save_graph/graph.png")

        if e % 100 == 0:
            agent.model.save_weights("save_model/model", save_format="tf")
