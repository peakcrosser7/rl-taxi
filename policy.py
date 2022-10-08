import numpy as np


class PolicyObj:
    def policy(self, state, **kwargs):
        """用于学习的策略函数"""
        pass

    def best_policy(self, state, **kwargs):
        """用于测试的最佳策略选择函数"""
        pass


class EpsilonGreedyPolicy(PolicyObj):
    """epsilon-greedy搜索策略"""

    def __init__(self, epsilon=0.01):
        """
        :param epsilon:探索概率
        """
        self.epsilon = epsilon

    def policy(self, state: int, **kwargs) -> np.intc:
        Q: np.ndarray = kwargs['Q']  # Q表(行为价值表)
        action_n: int = kwargs['action_n']  # 行为维度大小
        # # numpy.random.uniform():从0~1中随机采样
        if np.random.uniform() > self.epsilon:
            # 使用1-epsilon的概率选择最大行为价值
            # Q[state].argmax():返回Q表中state状态中值最大的元素对应的索引
            # 即,选择state状态中最大价值所对应的行为作为此次决策选择的行为
            action = Q[state].argmax()
        else:
            # 使用epsilon的概率在所有可选行为中随机选择
            action = np.random.randint(action_n)
        return action

    def best_policy(self, state: int, **kwargs) -> np.intc:
        Q: np.ndarray = kwargs['Q']
        # 直接选择Q表中当前状态最大价值所对应的行为
        return Q[state].argmax()
