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
        self.policy_obj: PolicyObj = policy_obj

        # # 衰减因子
        # self.gamma: float = gamma
        # # 学习速率参数alpha
        # self.learning_rate: float = learning_rate
        # # 动作维度
        # self.action_n: int = env.action_space.n

    def policy(self, train: bool, state, **kwargs):
        if train:
            return self.policy_obj.policy(state, **kwargs)
        else:
            return self.policy_obj.best_policy(state, **kwargs)

    def play(self, train, render=False) -> int:
        pass
        # episode_reward = 0
        # observation = self.env.reset()
        # while True:
        #     if render:
        #         self.env.render()
        #     action = self.decide(observation)
        #     next_observation, reward, done, _ = self.env.step(action)
        #     episode_reward += reward
        #     if train:
        #         self.learn(observation, action, reward, next_observation, done)
        #     if done:
        #         break
        #     observation = next_observation
        # return episode_reward

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

    def test(self):
        episode_rewards = [self.play(train=False) for _ in range(100)]
        print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                             len(episode_rewards), np.mean(episode_rewards)))
