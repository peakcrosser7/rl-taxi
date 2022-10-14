# 的士调度 Taxi-v4

import numpy as np

import config
from agents.agent import Agent
from agents.double_q_learning import DoubleQLearningAgent
from agents.expected_sarsa import ExpectedSarsaAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.sarsa_lambda import SarsaLambdaAgent
import env
from env import TaxiEnv
from policy import EpsilonGreedyPolicy

# 环境使用
np.random.seed(0)

env: TaxiEnv = env.get_sub_env(config.ENV_MAP, 5, 5, 1)
print('状态数量 = {}'.format(env.n_observation))
print('动作数量 = {}'.format(env.n_action))

# 环境的观测是一个[0,500)的整数
state = env.decode(env.reset())
# 观测可转换为4元组:出租车所在行和列,乘客位置,目标位置
print(state.taxi_row, state.taxi_col, state.pass_locs, state.dst_locs)
# 出租车所在行列取值范围为{0,1,2,3,4}
print('的士位置 = {}'.format((state.taxi_row, state.taxi_col)))
# 0~3表示在对应R,G,Y,B位置等待,4表示在车上
print('乘客位置 = {}'.format([env.locs[i] for i in state.pass_locs]))
# 0~3表示R,G,Y,B四个位置
print('目标位置 = {}'.format([env.locs[i] for i in state.dst_locs]))
env.render()

env.step(0)

env.render()

# 使用ε-贪婪策略
policy = EpsilonGreedyPolicy(epsilon=0.01)


def taxi(agent: Agent, train_times, test_times=100):
    print(agent.__class__.__name__ + ":")
    agent.train(train_times)
    agent.test(test_times)


taxi(SarsaAgent(env, policy), 3000, 100)

taxi(SarsaLambdaAgent(env, policy), 5000)

taxi(QLearningAgent(env, policy), 4000)

taxi(ExpectedSarsaAgent(env, policy), 5000)

taxi(DoubleQLearningAgent(env, policy), 9000)
