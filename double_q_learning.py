import numpy as np

from agent import Agent


class DoubleQLearningAgent(Agent):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        Agent.__init__(self, env, gamma, learning_rate, epsilon)
        self.q0: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))
        self.q1: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state) -> np.intc:
        if np.random.uniform() > self.epsilon:
            action = (self.q0 + self.q1)[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, **kwargs):
        if np.random.randint(2):
            self.q0, self.q1 = self.q1, self.q0
        a = self.q0[next_state].argmax()
        u = reward + self.gamma * self.q1[next_state, a] * (1. - done)
        td_error = u - self.q0[state, action]
        self.q0[state, action] += self.learning_rate * td_error
