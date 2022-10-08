import numpy as np

from agent import Agent


# Q 学习
class QLearningAgent(Agent):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        Agent.__init__(self, env, gamma, learning_rate, epsilon)
        self.q: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state) -> np.intc:
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, **kwargs):
        u = reward + self.gamma * self.q[next_state].max() * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error
