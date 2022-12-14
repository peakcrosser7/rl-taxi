import numpy as np

from agents.sarsa import SarsaAgent


class SarsaLambdaAgent(SarsaAgent):
    def __init__(self, env, policy_obj, lamda=0.6, gamma=0.9, learning_rate=0.1):
        SarsaAgent.__init__(self, env, policy_obj, gamma, learning_rate)
        self.lamda: float = lamda
        # E表(效用追迹表):状态维度(observation_space.n)×行为维度(action_space.n)大小的矩阵
        self.E: np.ndarray = np.zeros((env.n_observation, env.n_action))

    def learn(self, state, action, reward, next_state, next_action, done):
        # for e in E: E(s,a)=λ*γ*E(s,a)
        # 此处和伪代码不同,采用了先更新E的方式,因为初始E表全为0因此下句无效,从而不改变运算结果
        self.E *= (self.lamda * self.gamma)
        # E(S,A)=E(S,A)+1
        self.E[state, action] += 1.

        # δ=R+γ*Q(S',A')-Q(S,A)
        delta = reward + self.gamma * self.Q[next_state, next_action] * (1. - done) - self.Q[state, action]
        # for q in Q: Q(s,a)=Q(s,a)+α*δ*E(s,a)
        self.Q += self.learning_rate * self.E * delta
        if done:
            self.E *= 0.
