import gym
import numpy as np
from matplotlib import pyplot as plt

from policy import PolicyObj


class Agent:
    def __init__(self, env: gym.Env, policy_obj: PolicyObj):
        """
        :param env: 强化学习使用的游戏环境
        :param policy_obj: 策略方法对象
        """
        # 强化学习使用的游戏环境
        self.env: gym.Env = env
        # 使用的策略对象
        self.policy_obj: PolicyObj = policy_obj

    def policy(self, train: bool, state, **kwargs):
        if train:
            return self.policy_obj.policy(state, **kwargs)
        else:
            return self.policy_obj.best_policy(state, **kwargs)

    def play(self, train, render=False) -> int:
        pass

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

    def test(self, episodes: int = 100):
        episode_rewards = [self.play(train=False) for _ in range(episodes)]
        print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                             len(episode_rewards), np.mean(episode_rewards)))
