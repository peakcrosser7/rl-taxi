# # 的士调度 Taxi-v3
# %%
import gym
import numpy as np

# ### 环境使用
from double_q_learning import DoubleQLearningAgent
from expected_sarsa import ExpectedSARSAAgent
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from sarsa_lambda import SARSALambdaAgent

np.random.seed(0)

env: gym.Env = gym.make('Taxi-v3')
env.seed(0)
print('观察空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('状态数量 = {}'.format(env.observation_space.n))
print('动作数量 = {}'.format(env.action_space.n))

state = env.reset()
taxi_row, taxi_col, pass_loc, dest_idx = env.unwrapped.decode(state)
print(taxi_row, taxi_col, pass_loc, dest_idx)
print('的士位置 = {}'.format((taxi_row, taxi_col)))
print('乘客位置 = {}'.format(env.unwrapped.locs[pass_loc]))
print('目标位置 = {}'.format(env.unwrapped.locs[dest_idx]))
env.render()

env.step(0)

env.render()

agent = SARSAAgent(env)
agent.train(3000)
agent.test()

agent = ExpectedSARSAAgent(env)
agent.train(5000)
agent.test()

agent = SARSALambdaAgent(env)
agent.train(5000)
agent.test()

agent = QLearningAgent(env)
agent.train(4000)
agent.test()

agent = DoubleQLearningAgent(env)
agent.train(9000)
agent.test()

