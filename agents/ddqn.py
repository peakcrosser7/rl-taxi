import random
from collections import deque

import keras
import numpy as np
from keras.layers import Embedding, Reshape, Dense
from keras.optimizers import Adam

from env import TaxiEnv


class DDQNAgent:
    def __init__(self, env: TaxiEnv, weights_h5: str, is_weight=False,
                 gamma=0.95,
                 epsilon_min=0.001,
                 max_steps=1000
                 ):
        # 游戏环境
        self.env: TaxiEnv = env
        # 刚开始输入的字符串是否为可读取模型
        self.is_weight = is_weight
        # 网络模型参数文件路径
        self.weights_h5: str = weights_h5
        # 训练网络模型
        self.model = self._build_model()
        # 目标网络模型
        self.target_model = self._build_model()

        # 经验回放机制使用的经验池
        self.memory_buffer = deque(maxlen=2000)
        self.gamma: float = gamma
        self.epsilon_decay = 0.01
        self.epsilon_min = epsilon_min
        self.epsilon = self.epsilon_min if is_weight else 1
        # 设置最大步数
        self.env.set_max_steps(max_steps)

    def _build_model(self):
        """构建深度学习网络"""
        # 输入的维度,即环境可能的状态的维度,taxi-v3游戏中是1维
        input_length = 1
        action_size = self.env.n_action
        # 输入层:初始化深度学习网络输入层的tensor,输入是1为的向量
        ipt = keras.Input(shape=(input_length,), name="input")
        # 嵌入层:模型第一层,将正整数(下标)转换为具有固定大小的向量
        # input_dim:下标的范围,是taxi的状态数量500
        # output_dim:转换成的向量的维度,是全连接嵌入的维度
        # input_length:输入序列的长度
        # 对输入向量进行嵌入层处理:将500种1维状态转换为500种10维向量
        layer = Embedding(self.env.n_observation, 10, input_length=1)(ipt)
        # 重构层:将一定维度的多维矩阵重新构造为一个新的元素数量相同但是维度尺寸不同的矩阵
        # target_shape:目标尺寸为10
        # 对嵌入层输出重构,将原来500*1的10维向量矩阵其转换为500*10的矩阵
        layer = Reshape((10,))(layer)

        # 全连接层
        # units:输出空间维度,即神经元个数
        # activation:激活函数
        layer = Dense(50, activation="relu")(layer)
        layer = Dense(50, activation="relu")(layer)
        layer = Dense(50, activation="relu")(layer)
        # 输出全连接层: 输出空间维度为agent的行为的个数
        output = Dense(action_size, activation='linear')(layer)
        # 构建网络模型
        model = keras.Model(inputs=[ipt], outputs=[output])
        # 输出网络模型的信息状况
        model.summary()
        # 加载权重文件中的数据
        if self.is_weight:
            # 加载网络中所有层的权重数据
            model.load_weights(self.weights_h5)
        return model

    def _update_target_model(self):
        """更新目标网络模型的权重参数"""
        # 使用训练网络的权重更新目标网络
        self.target_model.set_weights(self.model.get_weights())

    def _remember(self, state: int, action: int, reward: int,
                  next_state: int, done: bool):
        """
        向经验池添加数据
        :param state: 状态
        :param action: 动作
        :param reward: 回报
        :param next_state: 下一个状态
        :param done: 游戏结束标志
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def process_batch(self, batch_size: int):
        """
        对一批次的采样数据进行处理
        :param batch_size: 一批次数据的个数
        :return:
            - states - 采样的当前状态数组
            - y - 训练网络经采样处理后得到的当前状态的行为价值表
        """
        # 从经验池中随机采样一个batch(批次)
        data = random.sample(self.memory_buffer, batch_size)

        # 生成Q_target
        # 从batch中提取出所有的当前状态和下一状态
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])
        # 训练网络当前状态预测的行为价值表
        y = self.model.predict(states)
        # 目标网络下一状态预测的行为价值表
        q = self.target_model.predict(next_states)

        # 遍历每个采样的数据
        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                # target更新采用Q-Learning的思想
                # r_j+γ*max_a'(Q'(S',a')
                target += self.gamma * np.amax(q[i])
            # target值由于有回报值的参与,因此相对准确,
            # 进而使用该值来更新训练网络对应状态的行为价值
            y[i][action] = target
        return states, y

    def _policy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 5)
        else:
            # 根据输入得到预测的数组
            q_values = self.model.predict([state])[0]
            return np.argmax(q_values)

    def _update_epsilon(self, episode):
        """更新epsilon"""
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)

    def train(self, episodes: int, batch_size: int, save_weights=True):
        # 配置网络以用于训练
        # loss:损失函数,mse:均方误差
        # optimizer:优化器,通过比较预测和损耗函数来优化输入权重
        self.model.compile(loss='mse', optimizer=Adam(1e-3))

        count = 0  # 训练次数

        # 输出文件列表
        file_train = open("train_output_history.csv", "w", encoding='utf-8')
        train_output_file = [["episodes", "reward_sum", "reward_sum_normalized", "loss", "epsilon", "step_times"]]
        row_txt = "{},{},{},{},{},{}".format("episodes", "reward_sum", "reward_sum_normalized", "loss", "epsilon",
                                             "step_times")
        file_train.write(row_txt)
        file_train.write('\n')
        file_train.close()

        for i in range(episodes):
            state = self.env.reset()
            # reward_sum = 0  # 一次游戏的总回报
            loss = np.infty  # 初始化为正无穷
            done = False
            step_times = 0  # 每次游戏走的步数
            while not done:

                # 通过贪婪选择法ε-greedy选择action
                action = self._policy(state)
                next_state, reward, done, _ = self.env.step(action)
                # 将数据加入到经验池
                self._remember(state, action, reward, next_state, done)

                # 当经验池值数据足够batch_size大小时进行训练
                if len(self.memory_buffer) > batch_size * 10:
                    # X:采样的状态数组,y:训练网络当前状态的行为价值表
                    X, y = self.process_batch(batch_size)
                    # 对数据进行训练
                    loss = self.model.train_on_batch(X, y)
                    step_times += 1
                    count += 1
                    # 固定次数更新目标网络
                    if count != 0 and count % batch_size == 0:
                        self._update_target_model()
                state = next_state

            # 减小epsilon-greedy的epsilon参数
            self._update_epsilon(i)

            reward_sum_normalized = self.env.normalized_total_reward()
            reward_sum = self.env.total_reward()

            # 将每一次训练历史写入文件列表
            train_output_file.append([i, reward_sum, loss, self.epsilon, step_times])

            # 每次训练完写入一行
            file_train = open("train_output_history.csv", "a", encoding='utf-8')
            row_txt = "{},{},{},{},{},{}".format(i, reward_sum, reward_sum_normalized, loss, self.epsilon, step_times)
            file_train.write(row_txt)
            file_train.write('\n')
            file_train.close()

            if i % 10 == 0:
                # 若总回报超过6返回
                # if reward_sum > 6:
                #   return history
                self.model.save_weights(self.weights_h5)
                print('episode: {} | episode reward: {:.2f} | loss: {:.4f} | epsilon:{:.2f}'
                      .format(i, reward_sum, loss, self.epsilon))

            if save_weights:
                self.model.save_weights(self.weights_h5)

        return train_output_file

    def test(self, episodes: int, render=False):
        """使用训练好的模型测试游戏"""

        episode_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            while True:
                if render:
                    self.env.render()
                # 训练网络当前状态预测得到的行为价值表
                q_values: np.ndarray = self.model.predict([int(state)])[0]
                # 取最大价值的动作
                action = int(np.argmax(q_values))
                state, _, done, _ = self.env.step(action)
                if done:
                    episode_rewards.append(self.env.normalized_total_reward())
                    break
        print('平均回合奖励 = {:.2f} / {} = {:.2f}'
              .format(sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))
