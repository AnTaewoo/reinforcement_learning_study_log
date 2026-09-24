import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DeepSARSA(tf.keras.Model):
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = Dense(30, activation="relu")
        self.fc2 = Dense(30, activation="relu")
        self.fc_out = Dense(action_size)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.fc_out(x)


class DeepSARSAagent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.lr = 0.001
        self.discount_factor = 0.99
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.action_size)
        self.optim = Adam(lr=self.lr)

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(0, self.action_size)
        else:
            q_lists = self.model(state)[0]
            return np.argmax(q_lists)

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)

            predict = self.model(state)[0]
            action = tf.one_hot([action], self.action_size)
            predict = tf.reduce_sum(action * predict, axis=1)

            next_q = self.model(next_state)[0][next_action]
            target = reward + self.discount_factor * next_q

            loss = tf.reduce_mean(tf.square(target - predict))
        grad = tape.gradient(loss, model_params)
        self.optim.apply_gradients(zip(grad, model_params))


if __name__ == "__main__":
    env = Env()
    state_size = 15
    action = [0, 1, 2, 3, 4]
    action_size = len(action)

    agent = DeepSARSAagent(state_size, action_size)

    scores, episodes = [], []

    EPISODE = 300
    for e in range(EPISODE):
        done = False
        score = 0
        state = env.reset()
        state = tf.reshape(state, [1, state_size])

        while not done:
            action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            next_state = tf.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            agent.train_model(state, action, reward, next_state, next_action, done)

            score += reward
            state = next_state

            if done:
                scores.append(score)
                episodes.append(e)

        if e % 50 == 0:
            agent.model.save_weights("save_model/model", save_format="tf")
