import sys
import gymnasium as gym
import pylab
import random
import numpy as np
import math
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
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
    def __init__(self, s_size, action_size):
        self.s_size = s_size
        self.action_size = action_size
        self.model = DQN(action_size)
        self.model.load_weights("./save_model/model")

    def get_action(self, s):
        q_list = self.model(s)
        return np.argmax(q_list)


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    env.unwrapped.metadata["render_fps"] = 1000
    env.unwrapped.theta_threshold_radians = math.radians(45)  # 실패 각도, 기본 12°
    env.unwrapped.x_threshold = 2.5
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    agent = DQNagent(s_size, a_size)

    EPISODE = 10
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

            score += r
            s = next_s

            if done:
                print("episode: {:3d} | score: {:.3f} ".format(e, score))
