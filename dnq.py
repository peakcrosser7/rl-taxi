import random
from collections import deque

import gym
import numpy as np
from keras.layers import Input, Dense, Reshape, Embedding
from keras.models import Model
from keras.optimizers import Adam


class DQN:
    def __init__(self, loadweight):
        self.loadweight = loadweight
        self.env = gym.make("Taxi-v3")
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.memory_buffer = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.001
        self.env._max_episode_steps = 5000000000
        if self.loadweight:
            self.epsilon = self.epsilon_min
        else:
            self.epsilon = 1

    def build_model(self):
        input_length = 1
        action_size = self.env.action_space.n
        input = Input(shape=(input_length,), name="input")
        layer = Embedding(500, 10, input_length=1)(input)
        layer = Reshape((10,))(layer)

        layer = Dense(50, activation="relu")(layer)
        layer = Dense(50, activation="relu")(layer)
        layer = Dense(50, activation="relu")(layer)

        output = Dense(action_size, activation='linear')(layer)
        model = Model(inputs=[input], outputs=[output])
        model.summary()
        if self.loadweight:
            model.load_weights(weightname)
        return model

    def update_target_model(self):
        """更新target_model
        """
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
            X: states
            y: [Q_value1, Q_value2]
        """
        # 从经验池中随机采样一个batch
        data = random.sample(self.memory_buffer, batch)

        # 生成Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])
        y = self.model.predict(states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target
        return states, y

    def train(self, episode, batch):
        """训练
        Arguments:
            episode: 游戏次数
            batch： batch size

        Returns:
            history: 训练记录
        """

        self.model.compile(loss='mse', optimizer=Adam(1e-3))

        history = {'episode': [], 'Episode_reward': [], 'Loss': []}

        count = 0
        for i in range(episode):
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False
            action_number = 0
            while not done and action_number < max_action_number:
                # 通过贪婪选择法ε-greedy选择action。
                x = observation
                action = self.egreedy_action(x)
                observation, reward, done, _ = self.env.step(action)
                # 将数据加入到经验池。
                reward_sum += reward
                self.remember(x, action, reward, observation, done)

                if len(self.memory_buffer) > batch:
                    # 训练
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)

                    count += 1

                    # 固定次数更新target_model
                    if count != 0 and count % batch == 0:
                        self.update_target_model()
                action_number += 1
            # 减小egreedy的epsilon参数。
            self.update_epsilon(i)
            if i % 5 == 0:
                history['episode'].append(i)
                history['Episode_reward'].append(reward_sum)
                history['Loss'].append(loss)
            if i % 10 == 0:
                if reward_sum > 6:
                    return history
                print('Episode: {} | Episode reward: {} | loss: {:.4f} | e:{:.2f}'.format(i, reward_sum, loss,
                                                                                          self.epsilon))
            self.model.save_weights(weightname)

        return history

    def play(self):
        """使用训练好的模型测试游戏.
        """

        observation = self.env.reset()

        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 1:

            self.env.render()

            x = observation
            q_values = self.model.predict([int(x)])[0]
            action = np.argmax(q_values)
            observation, reward, done, _ = self.env.step(action)

            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = self.env.reset()

        self.env.close()


if __name__ == '__main__':
    weightname = "dqn_Taxi-v3_weights.h5"
    model = DQN(True)
    max_action_number = 5000
    history = model.train(500, 16)
    model.play()
