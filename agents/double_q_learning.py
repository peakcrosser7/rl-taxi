import numpy as np

from agents.agent import Agent
from policy import GreedyPolicy


class DoubleQLearningAgent(Agent):
    def __init__(self, env, behave_policy, borrow_policy=GreedyPolicy, gamma=0.9, learning_rate=0.1):
        Agent.__init__(self, env, behave_policy)
        # 衰减因子
        self.gamma: float = gamma
        # 学习速率参数α
        self.learning_rate: float = learning_rate
        # 两个Q表
        self.Q0: np.ndarray = np.zeros((env.n_observation, env.n_action))
        self.Q1: np.ndarray = np.zeros((env.n_observation, env.n_action))
        # 借鉴策略对象
        self.borrow_policy = borrow_policy

    def behavioral_strategy(self, train: bool, state, Q):
        Q = Q if self.env.taxi_at_locs(state) else Q[:, :4]
        return self.policy(train, state, Q)

    def borrowing_strategy(self, state, Q):
        """
        借鉴策略
        使用贪婪策略
        """
        Q = Q if self.env.taxi_at_locs(state) else Q[:, :4]
        return self.borrow_policy.policy(state, Q)

    def learn(self, state, action, reward, next_state, done):
        # 1/2 概率交换两个Q表
        if np.random.randint(2):
            self.Q0, self.Q1 = self.Q1, self.Q0
        # a'=argmax(Q0(s_{t+1},a))
        a = self.borrowing_strategy(next_state, self.Q0)
        # u=R_{t+1}+γ*max_{a'}(Q1(S_{t+1},a'))
        u = reward + self.gamma * self.Q1[next_state, a] * (1. - done)
        # td_error=R_{t+1}+γ*max_{a'}(Q1(S_{t+1},a'))-Q0(S_t,A_t)
        td_error = u - self.Q0[state, action]
        # Q0(S_t,A_t)=Q0(S_t,A_t)+α(R_{t+1}+γ*max_{a'}(Q1(S_{t+1},a'))-Q0(S_t,A_t))
        self.Q0[state, action] += self.learning_rate * td_error

    def play(self, train=False, render=False) -> float:
        state = self.env.reset()
        while True:
            if render:
                self.env.render()
            action = self.behavioral_strategy(train, state, Q=self.Q0 + self.Q1)
            next_state, reward, done, _ = self.env.step(action)
            if train:
                self.learn(state, action, reward, next_state, done)
            if done:
                return self.env.normalized_total_reward()
            state = next_state
