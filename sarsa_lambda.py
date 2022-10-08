import numpy as np

from sarsa import SARSAAgent


class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, lamda=0.6, beta=1.,
                 gamma=0.9, learning_rate=0.1, epsilon=0.01):
        SARSAAgent.__init__(self, env, gamma=gamma, learning_rate=learning_rate,
                            epsilon=epsilon)
        self.lamda: float = lamda
        self.beta: float = beta
        self.e: np.ndarray = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done, **kwargs):
        next_action = kwargs['next_action']
        # 更新资格迹
        self.e *= (self.lamda * self.gamma)
        self.e[state, action] = 1. + self.beta * self.e[state, action]

        # 更新价值
        u = reward + self.gamma * \
            self.q[next_state, next_action] * (1. - done)
        td_error = u - self.q[state, action]
        self.q += self.learning_rate * self.e * td_error
        if done:
            self.e *= 0.
