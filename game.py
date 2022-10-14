import time

import keyboard

import config
from env import TaxiEnv


def game():
    taxi_env = TaxiEnv(config.ENV_MAP, config.PASS_LOCS, config.NUM_PASS)
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
        elif keyboard.is_pressed('s'):
            action = 4
        elif keyboard.is_pressed('x'):
            action = 5
        elif keyboard.is_pressed('q'):
            done = True
        if action is not None:
            taxi_env.step(action)
            taxi_env.render()
            time.sleep(0.3)
        if taxi_env.is_done():
            done = True


game()
