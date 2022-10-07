import numpy as np


class Agent:
    def __init__(self, env, gamma, learning_rate, epsilon):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        # self.q = np.zeros((env.observation_space.n, env.action_space.n))
        self.action_n = env.action_space.n

    def decide(self, state):
        pass

    def learn(self, state, action, reward, next_state, done, **kwargs):
        pass
