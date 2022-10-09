import numpy as np

from agents.agent import Agent
from policy import GreedyPolicy


class QLearningAgent(Agent):
    def __init__(self, env, behave_policy, borrow_policy=GreedyPolicy, gamma=0.9, learning_rate=0.1):
        Agent.__init__(self, env, behave_policy)
        # 衰减因子
        self.gamma: float = gamma
        # 学习速率参数α
        self.learning_rate: float = learning_rate
        # 动作维度
        self.action_n: int = env.action_space.n
        self.Q: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))
        # 借鉴策略对象
        self.borrow_policy = borrow_policy

    def behavioral_policy(self, train: bool, state, Q):
        """行为策略"""
        return self.policy(train, state, Q)

    def borrowing_policy(self, state, Q):
        """借鉴策略"""
        return self.borrow_policy.policy(state, Q)

    def learn(self, state, action, reward, next_state, done):
        # u=R_{t+1}+γ*max_{a'}(Q(S_{t+1},a'))
        u = reward + self.gamma * self.Q[next_state, self.borrowing_policy(next_state, self.Q)] * (1. - done)
        # td_error*R_{t+1}+γ*max_{a'}(Q(S_{t+1},a'))-Q(S_t,A_t)
        td_error = u - self.Q[state, action]
        # Q(S_t,A_t)=Q(S_t,A_t)+α*(R_{t+1}+γ*max_{a'}(Q(S_{t+1},a')-Q(S_t,A_t)))
        self.Q[state, action] += self.learning_rate * td_error

    def play(self, train=False, render=False) -> int:
        episode_reward = 0
        state = self.env.reset()
        while True:
            if render:
                self.env.render()
            action = self.behavioral_policy(train, state, self.Q)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if train:
                self.learn(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        return episode_reward
