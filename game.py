import time

import keyboard

import config
import env
from env import TaxiEnv


def game():
    seed = int(time.time_ns() % (2 ** 32))
    taxi_env: TaxiEnv = env.get_sub_env(config.ENV_MAP, 14, 14, 2, seed=seed)
    taxi_env.render()

    done = False
    while not done:
        action = None
        if keyboard.is_pressed('down'):
            action = 0
        elif keyboard.is_pressed('up'):
            action = 1
        elif keyboard.is_pressed('right'):
            action = 2
        elif keyboard.is_pressed('left'):
            action = 3
        if keyboard.is_pressed('s'):
            action = 4
        elif keyboard.is_pressed('x'):
            action = 5
        elif keyboard.is_pressed('q'):
            done = True
        if action is not None:
            taxi_env.step(action)
            taxi_env.render(sleep=0.)
            time.sleep(0.3)
        if taxi_env.is_done():
            done = True


game()
