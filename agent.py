import gym
import numpy as np
from matplotlib import pyplot as plt


class Agent:
    def __init__(self, env: gym.Env, gamma: float, learning_rate: float, epsilon: float):
        """
        :param env: 强化学习使用的游戏环境
        :param gamma: 衰减因子
        :param learning_rate: 学习速率参数
        :param epsilon: 探索概率
        """
        # 强化学习使用的游戏环境
        self.env: gym.Env = env
        # 衰减因子
        self.gamma: float = gamma
        # 学习速率参数alpha
        self.learning_rate: float = learning_rate
        # 探索概率,即epsilon-greedy搜索策略中的epsilon
        self.epsilon: float = epsilon
        # 动作维度
        self.action_n: int = env.action_space.n

    def decide(self, state) -> np.intc:
        """
        策略函数
        :param state:当前状态
        :return: 根据状态选择的行为
        """
        pass

    def learn(self, state, action, reward, next_state, done, **kwargs):
        pass

    def play(self, train=False, render=False) -> int:
        episode_reward = 0
        observation = self.env.reset()
        while True:
            if render:
                self.env.render()
            action = self.decide(observation)
            next_observation, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if train:
                self.learn(observation, action, reward, next_observation, done)
            if done:
                break
            observation = next_observation
        return episode_reward

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
        self.epsilon = 0.  # 取消探索

        episode_rewards = [self.play() for _ in range(100)]
        print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
                                             len(episode_rewards), np.mean(episode_rewards)))
