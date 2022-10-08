# 的士调度 Taxi-v3
import gym
import numpy as np
from double_q_learning import DoubleQLearningAgent
from expected_sarsa import ExpectedSarsaAgent
from policy import EpsilonGreedyPolicy
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from sarsa_lambda import SarsaLambdaAgent

# 环境使用
np.random.seed(0)

env: gym.Env = gym.make('Taxi-v3')
env.seed(0)
print('观察空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('状态数量 = {}'.format(env.observation_space.n))
print('动作数量 = {}'.format(env.action_space.n))

# 环境的观测是一个[0,500)的整数
state = env.reset()
# 观测可转换为4元组:出租车所在行和列,乘客位置,目标位置
taxi_row, taxi_col, pass_loc, dest_idx = env.unwrapped.decode(state)
print(taxi_row, taxi_col, pass_loc, dest_idx)
# 出租车所在行列取值范围为{0,1,2,3,4}
print('的士位置 = {}'.format((taxi_row, taxi_col)))
# 0~3表示在对应R,G,Y,B位置等待,4表示在车上
print('乘客位置 = {}'.format(env.unwrapped.locs[pass_loc]))
# 0~3表示R,G,Y,B四个位置
print('目标位置 = {}'.format(env.unwrapped.locs[dest_idx]))
env.render()

env.step(0)

env.render()

policy = EpsilonGreedyPolicy()

agent = SarsaAgent(env, policy)
agent.train(3000)
agent.test()

agent = ExpectedSarsaAgent(env, policy)
agent.train(5000)
agent.test()

agent = SarsaLambdaAgent(env, policy)
agent.train(5000)
agent.test()

agent = QLearningAgent(env, policy)
agent.train(4000)
agent.test()

agent = DoubleQLearningAgent(env, policy)
agent.train(9000)
agent.test()
