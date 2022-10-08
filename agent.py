import gym
import numpy as np
from matplotlib import pyplot as plt


class Agent:
    def __init__(self, env: gym.Env, gamma: float, learning_rate: float, epsilon: float):
        self.env: gym.Env = env
        self.gamma: float = gamma
        self.learning_rate: float = learning_rate
        self.epsilon: float = epsilon
        self.action_n: int = env.action_space.n

    def decide(self, state) -> np.intc:
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
