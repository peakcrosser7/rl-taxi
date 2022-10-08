import numpy as np

from agent import Agent


# SARSA 算法

class SarsaAgent(Agent):
    """Sarsa算法智能体"""

    def __init__(self, env, gamma=0.9, learning_rate=0.2, epsilon=0.01):
        Agent.__init__(self, env, gamma, learning_rate, epsilon)

        # Q表(行为价值表):状态维度(observation_space.n)×行为维度(action_space.n)大小的矩阵
        # 每个元素表示状态行为对(S,A)的价值
        # 初始设置为全0
        self.Q: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))

    def decide(self, state) -> np.intc:
        # 基于epsilon-greedy搜索策略进行决策

        # numpy.random.uniform():从0~1中随机采样
        if np.random.uniform() > self.epsilon:
            # 使用1-epsilon的概率选择最大行为价值
            # q[state].argmax():返回Q表中state状态中值最大的元素对应的索引
            # 即,选择state状态中最大价值所对应的行动作为此次决策选择的行动
            action = self.Q[state].argmax()
        else:
            # 使用epsilon的概率在所有可选行为中随机选择
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, **kwargs):
        # 根据epsilon-贪婪策略进行策略迭代
        # Q(S,A)=Q(S-A)+alpha(R+gamma*Q(S',A')-Q(S,A))
        next_action = kwargs['next_action']
        # u=R+gamma*Q(S',A')
        u = reward + self.gamma * \
            self.Q[next_state, next_action] * (1. - done)
        # td_error=R+gamma*Q(S',A')-Q(S,A)
        td_error = u - self.Q[state, action]
        # Q(S,A)+=alpha(R+gamma*Q(S',A')-Q(S,A))
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
        action: np.intc = self.decide(state)
        while True:
            if render:
                self.env.render()
            # 执行一次动作,并得到新观察到的状态,奖励值,是否完成游戏
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            # 根据新的状态进行决策(终止状态时此步无意义)
            next_action = self.decide(next_state)
            if train:
                self.learn(state, action, reward, next_state, done, next_action=next_action)
            if done:
                break
            state, action = next_state, next_action
        return episode_reward
