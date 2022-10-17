import numpy as np

from agents.agent import Agent
from policy import EpsilonGreedyPolicy


class ExpectedSarsaAgent(Agent):
    def __init__(self, env, policy_obj: EpsilonGreedyPolicy, gamma=0.9, learning_rate=0.1):
        Agent.__init__(self, env, policy_obj)
        # ε-greedy的探索概率,用于求期望
        self.epsilon = policy_obj.epsilon
        # 衰减因子
        self.gamma: float = gamma
        # 学习速率参数α
        self.learning_rate: float = learning_rate
        # Q表
        self.Q: np.ndarray = np.zeros((env.n_observation, env.n_action))

    def learn(self, state, action, reward, next_state, done):
        Q = self.Q if self.env.taxi_at_locs(next_state) else self.Q[:, :4]
        # 使用下一状态行为价值Q(S_{t+1},A')的期望E来更新TD目标
        # E_π[Q(S_{t+1},A_{t+1}|S_{t+1}]=sum_a(π(a|S_{t+1})Q(S_{t+1},a))
        # =ε*avg(Q(s,a))+(1-ε)*max(Q(s,a))
        E = self.epsilon * Q[next_state].mean() + (1. - self.epsilon) * Q[next_state].max()
        # u=R_{t+1}+γ*E
        u = reward + self.gamma * E * (1. - done)
        # td_error=R_{t+1}+γ*E-Q(S_t,A_t)
        td_error = u - self.Q[state, action]
        # Q(S_t,A_t)=Q(S_t,A_t)+α(R_{t+1}+γ*E-Q(S_t,A_t))
        self.Q[state, action] += self.learning_rate * td_error

    def play(self, train=False, render=False) -> float:
        state = self.env.reset()
        while True:
            if render:
                self.env.render()
            Q = self.Q if self.env.taxi_at_locs(state) else self.Q[:, :4]
            action = self.policy(train, state, Q)
            next_state, reward, done, _ = self.env.step(action)
            if train:
                self.learn(state, action, reward, next_state, done)
            if done:
                return self.env.normalized_total_reward()
            state = next_state
