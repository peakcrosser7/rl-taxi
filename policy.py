from abc import abstractmethod

import numpy as np


class PolicyObj:
    @abstractmethod
    def policy(self, state, Q):
        """用于学习的策略函数"""
        raise NotImplementedError


class GreedyPolicy(PolicyObj):
    @classmethod
    def policy(cls, state: int, Q: np.ndarray):
        return Q[state].argmax()


class EpsilonGreedyPolicy(PolicyObj):
    """ε-greedy搜索策略"""

    def __init__(self, epsilon=0.01, epsilon_min=0.01, epsilon_decay=0.01):
        """
        :param epsilon:探索概率
        """
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def policy(self, state: int, Q: np.ndarray) -> np.intc:
        # # numpy.random.uniform():从0~1中随机采样
        if np.random.uniform() > self.epsilon:
            # 使用1-ε的概率选择最大行为价值
            # Q[state].argmax():返回Q表中state状态中值最大的元素对应的索引
            # 即,选择state状态中最大价值所对应的行为作为此次决策选择的行为
            action = Q[state].argmax()
        else:
            # 使用ε的概率在所有可选行为中随机选择
            action = np.random.randint(Q.shape[1])
        return action

    def update_epsilon(self, episode):
        """更新epsilon
        """
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)
