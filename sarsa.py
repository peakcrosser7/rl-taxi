import numpy as np

from agent import Agent


# SARSA 算法
class SARSAAgent(Agent):
    def __init__(self, env, gamma=0.9, learning_rate=0.2, epsilon=0.01):
        Agent.__init__(self, env, gamma, learning_rate, epsilon)
        self.q: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state) -> np.intc:
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, **kwargs):
        next_action = kwargs['next_action']
        u = reward + self.gamma * \
            self.q[next_state, next_action] * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learning_rate * td_error

    def play(self, train=False, render=False) -> int:
        episode_reward = 0
        observation = self.env.reset()
        action: np.intc = self.decide(observation)
        while True:
            if render:
                self.env.render()
            next_observation, reward, done, _ = self.env.step(action)
            episode_reward += reward
            next_action = self.decide(next_observation)  # 终止状态时此步无意义
            if train:
                self.learn(observation, action, reward, next_observation, done, next_action=next_action)
            if done:
                break
            observation, action = next_observation, next_action
        return episode_reward
