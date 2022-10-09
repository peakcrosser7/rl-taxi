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
        # 动作维度
        self.action_n: int = env.action_space.n
        # Q表
        self.Q: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done):
        # 使用下一状态行为价值Q(S_{t+1},A')的期望E来更新TD目标
        # E_π[Q(S_{t+1},A_{t+1}|S_{t+1}]=sum_a(π(a|S_{t+1})Q(S_{t+1},a))
        # =ε*avg(Q(s,a))+(1-ε)*max(Q(s,a))
        E = self.epsilon * self.Q[next_state].mean() + (1. - self.epsilon) * self.Q[next_state].max()
        # u=R_{t+1}+γ*E
        u = reward + self.gamma * E * (1. - done)
        # td_error=R_{t+1}+γ*E-Q(S_t,A_t)
        td_error = u - self.Q[state, action]
        # Q(S_t,A_t)=Q(S_t,A_t)+α(R_{t+1}+γ*E-Q(S_t,A_t))
        self.Q[state, action] += self.learning_rate * td_error

    def play(self, train=False, render=False) -> int:
        episode_reward = 0
        state = self.env.reset()
        while True:
            if render:
                self.env.render()
            action = self.policy(train, state, self.Q)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if train:
                self.learn(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        return episode_reward
