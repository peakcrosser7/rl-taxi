import time

import keyboard

from env import TaxiEnv


def game():
    env_map = [
        "+-----------------------------+",
        "|R: : : | | : | : : | : | : : |",
        "| : | : | : : | : : | : |G: : |",
        "| : | : | : : : : : | : : : | |",
        "| : | : : : | : | : | : : : | |",
        "| : : : | : | : : : : : | : : |",
        "| : : | | : | : | : : : | | : |",
        "| | : | : : : | | : | : : | : |",
        "| | : : | | : | | : | : : | | |",
        "| : | : : | : : : : : : | : : |",
        "| : | : : | : : | : : | : | : |",
        "| | : | : : | : : | : | : : | |",
        "| : : | | : : : | : | : : | : |",
        "| : : : | : | : | | :B: | : : |",
        "| : | : : | : | : | : : | | : |",
        "|Y| : : : | : | : : : | : | : |",
        "+-----------------------------+"
    ]

    pass_locs = {'R': (0, 0), 'G': (1, 12), 'B': (12, 10), 'Y': (14, 0)}
    taxi_env = TaxiEnv(env_map, pass_locs, 2)
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
        elif keyboard.is_pressed('s'):  # if key 'enter' is pressed
            action = 4
        elif keyboard.is_pressed('x'):
            action = 5
        elif keyboard.is_pressed('q'):
            done = True
        if action is not None:
            taxi_env.step(action)
            taxi_env.render()
            time.sleep(1)
        if taxi_env.is_done():
            done = True


game()
