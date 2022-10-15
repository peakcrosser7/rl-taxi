import random
from collections import deque

import numpy as np
from keras.layers import Input, Dense, Reshape, Embedding
from keras.models import Model
from keras.optimizers import Adam

import config
import env
from env import TaxiEnv


class DQN:
    def __init__(self, load_weight):
        self.load_weight = load_weight
        self.env: TaxiEnv = env.get_sub_env(config.ENV_MAP, 5, 5, 1)
        # 训练网络模型
        self.model = self.build_model()
        # 目标网络模型
        self.target_model = self.build_model()
        # 更新目标网络模型的权重参数
        self.update_target_model()

        # 经验回放机制使用的经验池
        self.memory_buffer = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.001
        self.env._max_episode_steps = 5000000000
        if self.load_weight:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon = 1

    def build_model(self):
        """构建深度学习网络"""
        # 输入的维度,即环境可能的状态的维度,taxi-v3游戏中是1维
        input_length = 1
        action_size = self.env.n_action
        # 输入层:初始化深度学习网络输入层的tensor,输入是1为的向量
        input = Input(shape=(input_length,), name="input")
        # 嵌入层:模型第一层,将正整数(下标)转换为具有固定大小的向量
        # input_dim:下标的范围,是taxi的状态数量500
        # output_dim:转换成的向量的维度,是全连接嵌入的维度
        # input_length:输入序列的长度
        # 对输入向量进行嵌入层处理:将500种1维状态转换为500种10维向量
        layer = Embedding(self.env.n_observation, 10, input_length=1)(input)
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
        model = Model(inputs=[input], outputs=[output])
        # 输出网络模型的信息状况
        model.summary()
        # 加载权重文件中的数据
        if self.load_weight:
            # 加载网络中所有层的权重数据
            model.load_weights(weight_name)
        return model

    def update_target_model(self):
        """更新目标网络模型的权重参数"""
        # 使用训练网络的权重到目标网络
        self.target_model.set_weights(self.model.get_weights())

    def egreedy_action(self, state):
        """ε-greedy选择action

        Arguments:
            state: 状态

        Returns:
            action: 动作
        """

        if np.random.rand() <= self.epsilon:
            return random.randint(0, 5)
        else:
            # 根据输入得到预测的数组
            q_values = self.model.predict([int(state)])[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """向经验池添加数据

        Arguments:
            state: 状态
            action: 动作
            reward: 回报
            next_state: 下一个状态
            done: 游戏结束标志
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self, episode):
        """更新epsilon
        """
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon = self.epsilon_min + (1 - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)

    def process_batch(self, batch):
        """batch数据处理

        Arguments:
            batch: batch size

        Returns:
            X: states 采样的当前状态数组
            y: [Q_value1, Q_value2] 经采样处理后得到的训练网络当前状态的行为价值表
        """
        # 从经验池中随机采样一个batch(批次)
        data = random.sample(self.memory_buffer, batch)

        # 生成Q_target
        # 从batch中提取出所有的当前状态和下一状态
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])
        # 训练网络当前状态预测得到的行为价值表
        y = self.model.predict(states)
        # 目标网络下一状态预测得到的行为价值表
        q = self.target_model.predict(next_states)

        # 遍历每个采样的数据
        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                # target更新采用Q-Learning的思想
                # r_j+γ*max_a'(Q'(S',a')
                target += self.gamma * np.amax(q[i])
            # 更新训练网络当前状态的行为价值
            y[i][action] = target
        return states, y

    def train(self, episode, batch_size):
        """训练
        Arguments:
            episode: 游戏次数
            batch_size： batch size

        Returns:
            history: 训练记录
        """

        # 配置网络以用于训练
        # loss:损失函数,mse:均方误差
        # optimizer:优化器,通过比较预测和损耗函数来优化输入权重
        self.model.compile(loss='mse', optimizer=Adam(1e-3))

        # 每训练5次后的历史
        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0  # 训练次数
        for i in range(episode):
            state = self.env.reset()
            reward_sum = 0  # 一次游戏的总回报
            loss = np.infty  # 初始化为正无穷
            done = False
            action_number = 0  # 一次游戏中行为的总次数
            while not done and action_number < max_action_number:
                # 通过贪婪选择法ε-greedy选择action。
                x = state
                action = self.egreedy_action(x)
                state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                # 将数据加入到经验池
                self.remember(x, action, reward, state, done)

                # 当经验池值数据足够batch_size大小时进行训练
                if len(self.memory_buffer) > batch_size:
                    # X:采样的状态数组,y:训练网络当前状态的行为价值表
                    X, y = self.process_batch(batch_size)
                    # 对数据进行训练
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    # 固定次数更新目标网络
                    if count != 0 and count % batch_size == 0:
                        self.update_target_model()
                action_number += 1
            # 减小epsilon-greedy的epsilon参数
            self.update_epsilon(i)
            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)
            if i % 10 == 0:
                # 若总回报超过6返回
                if reward_sum > 6:
                    return history
                print('Episode: {} | Episode reward: {} | loss: {:.4f} | e:{:.2f}'.format(i, reward_sum, loss,
                                                                                          self.epsilon))
            self.model.save_weights(weight_name)

        return history

    def play(self):
        """使用训练好的模型测试游戏.
        """

        state = self.env.reset()

        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 1:

            self.env.render()

            x = state
            # 训练网络当前状态预测得到的行为价值表
            q_values: np.ndarray = self.model.predict([int(x)])[0]
            # 取最大价值的动作
            action = int(np.argmax(q_values))
            state, reward, done, _ = self.env.step(action)

            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                state = self.env.reset()


if __name__ == '__main__':
    weight_name = "dqn_Taxi-v3_weights.h5"
    model = DQN(True)
    max_action_number = 5000
    history = model.train(500, 16)
    model.play()
