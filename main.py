# 的士调度 Taxi-v4
import time

import numpy as np

import config
import env
from agents.agent import Agent
from agents.double_q_learning import DoubleQLearningAgent
from agents.expected_sarsa import ExpectedSarsaAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from env import TaxiEnv
from policy import EpsilonGreedyPolicy

# 环境使用
np.random.seed(0)

taxi_env: TaxiEnv = env.get_sub_env(config.ENV_MAP, 6, 6, 1)
print('状态数量 = {}'.format(taxi_env.n_observation))
print('动作数量 = {}'.format(taxi_env.n_action))

# 环境的观测是一个[0,500)的整数
state = taxi_env.decode(taxi_env.reset())
# 观测可转换为4元组:出租车所在行和列,乘客位置,目标位置
print(state.taxi_row, state.taxi_col, state.pass_locs, state.dst_locs)
# 出租车所在行列取值范围为{0,1,2,3,4}
print('的士位置 = {}'.format((state.taxi_row, state.taxi_col)))
# 0~3表示在对应R,G,Y,B位置等待,4表示在车上
print('乘客位置 = {}'.format([taxi_env.locs[i] for i in state.pass_locs]))
# 0~3表示R,G,Y,B四个位置
print('目标位置 = {}'.format([taxi_env.locs[i] for i in state.dst_locs]))
taxi_env.render()

taxi_env.step(0)

taxi_env.render()

# 使用ε-贪婪策略
policy = EpsilonGreedyPolicy(epsilon=0.01)


def taxi(agent: Agent, train_times=10000, test_times=100):
    print(agent.__class__.__name__ + ":")
    agent.train(train_times)
    agent.test(test_times)


# seed = int(time.time_ns() % (2 ** 32))
# taxi(SarsaAgent(env.get_sub_env(config.ENV_MAP, 5, 5, 1, seed=seed), policy))
# taxi(SarsaAgent(env.get_sub_env(config.ENV_MAP, 6, 6, 1, seed=seed), policy))
# taxi(SarsaAgent(env.get_sub_env(config.ENV_MAP, 7, 7, 1, seed=seed), policy))
# taxi(SarsaAgent(env.get_sub_env(config.ENV_MAP, 8, 8, 1, seed=seed), policy))
# taxi(SarsaAgent(env.get_sub_env(config.ENV_MAP, 10, 10, 1, seed=seed), policy))
# taxi(SarsaAgent(env.get_sub_env(config.ENV_MAP, 15, 15, 1, seed=seed), policy))

taxi(SarsaAgent(taxi_env, policy))

taxi(SarsaLambdaAgent(taxi_env, policy))

taxi(QLearningAgent(taxi_env, policy))

taxi(ExpectedSarsaAgent(taxi_env, policy))

taxi(DoubleQLearningAgent(taxi_env, policy))
