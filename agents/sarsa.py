import numpy as np

from agents.agent import Agent


# SARSA 算法

class SarsaAgent(Agent):
    """Sarsa算法智能体"""

    def __init__(self, env, policy_obj, gamma=0.9, learning_rate=0.2):
        Agent.__init__(self, env, policy_obj)

        # 衰减因子
        self.gamma: float = gamma
        # 学习速率参数α
        self.learning_rate: float = learning_rate
        # 动作维度
        self.action_n: int = env.n_action
        # Q表(行为价值表):状态维度(observation_space.n)×行为维度(action_space.n)大小的矩阵
        # 每个元素表示状态行为对(S,A)的价值
        # 初始设置为全0
        self.Q: np.ndarray = np.zeros((env.n_observation, env.n_action))

    def learn(self, state, action, reward, next_state, next_action, done):
        # 根据ε-贪婪策略进行策略迭代
        # Q(S,A)=Q(S-A)+α(R+γ*Q(S',A')-Q(S,A))
        # u=R+γ*Q(S',A')
        u = reward + self.gamma * self.Q[next_state, next_action] * (1. - done)
        # td_error=R+γ*Q(S',A')-Q(S,A)
        td_error = u - self.Q[state, action]
        # Q(S,A)+=α*(R+γ*Q(S',A')-Q(S,A))
        self.Q[state, action] += self.learning_rate * td_error

    def play(self, train=False, render=False) -> int:
        """
        智能体与环境交互(直到游戏结束)
        :param train:是否为训练
        :param render:是否渲染执行步骤对应图像
        :return:本次交互(直到游戏结束)所获取的总奖励值
        """
        # 本次游戏所获得的总奖励值
        episode_reward = 0
        # 初始状态
        state = self.env.reset()
        # 初始行为:此时因初始Q表全为0而选择第一个行为
        action: np.intc = self.policy(train, state, self.Q)
        while True:
            if render:
                self.env.render()
            # 执行一次动作,并得到新观察到的状态,奖励值,是否完成游戏
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            # 根据新的状态进行决策(终止状态时此步无意义)
            next_action = self.policy(train, next_state, self.Q)
            if train:
                self.learn(state, action, reward, next_state, next_action, done)
            if done:
                break
            state, action = next_state, next_action
        return episode_reward
