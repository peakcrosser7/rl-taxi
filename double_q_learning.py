import numpy as np

from agent import Agent


class DoubleQLearningAgent(Agent):
    def __init__(self, env, policy_obj, gamma=0.9, learning_rate=0.1):
        Agent.__init__(self, env, policy_obj)
        # 衰减因子
        self.gamma: float = gamma
        # 学习速率参数alpha
        self.learning_rate: float = learning_rate
        # 动作维度
        self.action_n: int = env.action_space.n
        self.Q0: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))
        self.Q1: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done, **kwargs):
        if np.random.randint(2):
            self.Q0, self.Q1 = self.Q1, self.Q0
        a = self.Q0[next_state].argmax()
        u = reward + self.gamma * self.Q1[next_state, a] * (1. - done)
        td_error = u - self.Q0[state, action]
        self.Q0[state, action] += self.learning_rate * td_error

    def play(self, train=False, render=False) -> int:
        episode_reward = 0
        state = self.env.reset()
        while True:
            if render:
                self.env.render()
            action = self.policy(state, train, Q=self.Q0 + self.Q1, action_n=self.action_n)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if train:
                self.learn(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        return episode_reward
