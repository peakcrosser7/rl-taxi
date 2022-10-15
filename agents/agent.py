from abc import abstractmethod

import gym
import numpy as np
from matplotlib import pyplot as plt

from env import TaxiEnv
from policy import GreedyPolicy


class Agent:
    def __init__(self, env: TaxiEnv, policy_obj):
        """
        :param env: 强化学习使用的游戏环境
        :param policy_obj: 策略方法对象
        """
        # 强化学习使用的游戏环境
        self.env: TaxiEnv = env
        # 使用的策略对象
        self.policy_obj = policy_obj

    def policy(self, train: bool, state, Q):
        if train:
            return self.policy_obj.policy(state, Q)
        else:
            return GreedyPolicy.policy(state, Q)

    @abstractmethod
    def play(self, train, render=False):
        raise NotImplementedError

    def train(self, episodes: int):
        """
        进行训练
        :param episodes: 训练的局数
        :return:
        """
        episode_rewards = []
        for episode in range(episodes):
            episode_reward = self.play(train=True)
            episode_rewards.append(episode_reward)

        plt.plot(episode_rewards)
        plt.show()

    def test(self, episodes: int = 100, render=False):
        episode_rewards = [self.play(train=False, render=render) for _ in range(episodes)]
        print('平均回合奖励 = {:.2f} / {} = {:.2f}'
              .format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))
