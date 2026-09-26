import gymnasium as gym
import tensorflow as tf
import numpy as np
import math
import os
import sys
import pylab
import random
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform


class DQN(tf.keras.Model):
    def __init__(self, len_a):
        super(DQN, self).__init__()
        self.fc1 = Dense(25, activation="relu")
        self.fc2 = Dense(25, activation="relu")
        self.fout = Dense(len_a)  # activation="softmax"

    def call(self, s):
        x = self.fc1(s)
        x = self.fc2(x)
        return self.fout(x)


class DQNagent:
    def __init__(self, len_s, len_a):
        self.len_s = len_s
        self.len_a = len_a

        self.discount_v = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000  # replay memory 가 500 이전까지는 학습 x
        self.lr = 0.001

        self.replay_memory = deque(maxlen=3000)

        self.model = DQN(len_a)
        self.target_network = DQN(len_a)
        self.sync_step = 1000
        self.optimizer = Adam(learning_rate=self.lr)

    def update_target_model(self):
        self.target_network.set_weights(self.model.get_weights())  # theta' = theta

    def get_action(self, s):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.len_a)
        else:
            q_list = self.model(s)
            return np.argmax(q_list)

    def append_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state[0], action, reward, next_state[0], done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        batch = random.sample(self.replay_memory, self.batch_size)
        states = np.array([sample[0] for sample in batch])
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch])
        dones = np.array([sample[4] for sample in batch])

        model_params = self.model.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(model_params)

            predicts = self.model(states)
            actions = tf.one_hot(actions, self.len_a)
            predicts = tf.reduce_sum(actions * predicts, axis=1)

            target_predicts = self.target_network(next_states)
            maxqs = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_v * maxqs
            MSE = tf.reduce_mean(tf.math.square(targets - predicts))

        grads = tape.gradient(MSE, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    env.unwrapped.metadata["render_fps"] = 2000
    env.unwrapped.theta_threshold_radians = math.radians(45)  # 실패 각도, 기본 12°
    env.unwrapped.x_threshold = 2.5
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    agent = DQNagent(s_size, a_size)

    scores, episodes = [], []
    score_avg = 0
    score_max = 0

    EPISODE = 250
    for e in range(EPISODE):
        done = False
        score = 0

        s, _ = env.reset()
        s = np.reshape(s, [1, s_size])
        while not done:
            env.render()

            a = agent.get_action(s)
            next_s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            next_s = np.reshape(next_s, [1, s_size])
            agent.append_replay_memory(s, a, r, next_s, done)

            score += r
            r = 0.1 if not done else -1

            if len(agent.replay_memory) >= agent.train_start:
                agent.train_model()

            s = next_s

            if done:
                agent.update_target_model()
                score_avg = 0.9 * score_avg + 0.1 * score

                scores.append(score_avg)
                episodes.append(e)
                print(
                    f"episode {e} | score {score} | memory_length {len(agent.replay_memory)} | epsilon {agent.epsilon} | score_max {score_max}"
                )

                pylab.plot(episodes, scores, "b")
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("./save_graph/graph.png")

            if e >= 100 and score_max < score:
                score_max = score
                agent.model.save_weights("save_model/model", save_format="tf")
