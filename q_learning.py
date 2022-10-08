import numpy as np

from agent import Agent


# Q 学习
class QLearningAgent(Agent):
    def __init__(self, env, policy_obj, gamma=0.9, learning_rate=0.1):
        Agent.__init__(self, env, policy_obj)
        # 衰减因子
        self.gamma: float = gamma
        # 学习速率参数alpha
        self.learning_rate: float = learning_rate
        # 动作维度
        self.action_n: int = env.action_space.n
        self.Q: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done):
        u = reward + self.gamma * self.Q[next_state].max() * (1. - done)
        td_error = u - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

    def play(self, train=False, render=False) -> int:
        episode_reward = 0
        state = self.env.reset()
        while True:
            if render:
                self.env.render()
            action = self.policy(train, state, Q=self.Q, action_n=self.action_n)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if train:
                self.learn(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        return episode_reward
