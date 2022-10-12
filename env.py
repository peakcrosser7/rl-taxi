import copy
import os
import platform
import sys
from enum import Enum
from typing import List, Tuple, Dict

import numpy as np
from gym import utils


class TaxiEnv:
    class EnumAction(Enum):
        DOWN = 0
        UP = 1
        RIGHT = 2
        LEFT = 3
        PICK_UP = 4
        DROP_OFF = 5

    class State:
        def __init__(self):
            self.taxi_row: int = 0
            self.taxi_col: int = 0
            self.pass_idxes: List[int] = []
            self.dst_idxes: List[int] = []

    def __init__(self, map_str: List[str],
                 locs_map: Dict[str, Tuple[int, int]],
                 num_pass: int,
                 max_steps=200
                 ):
        self.desc = np.asarray(map_str, dtype="c")
        self.locs = []
        for w, (r, c) in locs_map.items():
            self.locs.append((r, c))
            self.desc[r + 1, c * 2 + 1] = w[0]
        locs = self.locs
        self.num_pass = num_pass
        self.max_steps = max_steps

        self.num_rows = num_rows = len(map_str) - 2  # 去除最上最下两行
        self.num_cols = num_cols = int((len(map_str[0]) - 1) / 2)
        self.num_locs = num_locs = len(locs)
        self.action_space = self.EnumAction.__len__()
        self.observation_space = num_rows * num_cols * (num_locs + 1) ** num_pass * num_locs ** num_pass

        self._elapsed_steps = 0
        self._total_reward = 0
        self._done = False
        self._origin_state = self._init_state()
        self._current_state = copy.deepcopy(self._origin_state)
        self._last_action = None

    def _init_state(self) -> State:
        loop = True
        while loop:
            i = np.random.randint(self.observation_space)
            state = self.decode(i)
            taxi_loc = (state.taxi_row, state.taxi_col)
            loop = False
            if taxi_loc in self.locs:
                loop = True
                continue
            for i in range(self.num_pass):
                if state.pass_idxes[i] == state.dst_idxes[i] \
                        or self._in_taxi(state.pass_idxes[i]):
                    loop = True
                    break
            if not loop:
                return state

    def reset(self):
        self._elapsed_steps = 0
        self._total_reward = 0
        self._origin_state = self._init_state()
        self._current_state = copy.deepcopy(self._origin_state)
        self._last_action = None

    def encode(self, state: State):
        opt = state.taxi_row
        opt *= self.num_cols
        opt += state.taxi_col
        for i in range(self.num_pass):
            opt *= (self.num_locs + 1)
            opt += state.pass_idxes[i]
        for i in range(self.num_pass):
            opt *= self.num_locs
            opt += state.dst_idxes[i]
        return opt

    def decode(self, ipt: int):
        assert 0 <= ipt < self.observation_space
        state = self.State()
        dst_locs = []
        for i in range(self.num_pass):
            dst_locs.append(int(ipt % self.num_locs))
            ipt = ipt // self.num_locs
        dst_locs.reverse()
        state.dst_idxes = dst_locs

        pass_locs = []
        for i in range(self.num_pass):
            pass_locs.append(int(ipt % (self.num_locs + 1)))
            ipt //= (self.num_locs + 1)
        pass_locs.reverse()
        state.pass_idxes = pass_locs

        state.taxi_col = int(ipt % self.num_cols)
        ipt //= self.num_cols
        state.taxi_row = int(ipt)
        return state

    def _in_taxi(self, pass_idx: int):
        return pass_idx == self.num_locs

    def _get_in_taxi(self, pass_idxes: List[int], i: int):
        pass_idxes[i] = self.num_pass

    @staticmethod
    def _get_off_taxi(pass_idxes: List[int], i: int, new_loc_idx: int):
        pass_idxes[i] = new_loc_idx

    def _num_delivered(self, state: State) -> int:
        """送达目的地的乘客数"""
        delivered = 0
        for i in range(self.num_pass):
            pass_idx = state.pass_idxes[i]
            if self._in_taxi(pass_idx):
                delivered += 1
        return delivered

    def _step(self, action: int) -> Tuple[int, int, bool, dict]:
        state = self._current_state
        reward = -1
        if self._done:
            return int(self.encode(state)), reward, self._done, {}

        taxi_loc = (row, col) = (state.taxi_row, state.taxi_col)

        act = self.EnumAction(action)
        if act == self.EnumAction.DOWN:
            state.taxi_row = min(row + 1, self.num_rows - 1)
        elif act == self.EnumAction.UP:
            state.taxi_row = max(row - 1, 0)
        elif act == self.EnumAction.RIGHT and self.desc[1 + row, 2 * col + 2] == b":":
            state.taxi_col = min(col + 1, self.num_cols - 1)
        elif act == self.EnumAction.LEFT and self.desc[1 + row, 2 * col] == b":":
            state.taxi_col = max(col - 1, 0)
        elif act == self.EnumAction.PICK_UP:
            match = False
            for i in range(self.num_pass):
                pass_idx = state.pass_idxes[i]
                if (not self._in_taxi(pass_idx)) and taxi_loc == self.locs[pass_idx]:
                    self._get_in_taxi(state.pass_idxes, i)
                    reward += 20
                    match = True
                    break  # 一步只接一个人
            if not match:
                reward -= 10
        elif act == self.EnumAction.DROP_OFF:
            match = False
            for i in range(self.num_pass):
                pass_idx = state.pass_idxes[i]
                dst_idx = state.dst_idxes[i]
                if taxi_loc == self.locs[dst_idx] and self._in_taxi(pass_idx):
                    self._get_off_taxi(state.pass_idxes, i, dst_idx)
                    reward += 40
                    match = True
                    if self._num_delivered(state) == self.num_pass:
                        self._done = True
                    break

            if not match and (taxi_loc in self.locs):
                in_taxi = []
                for i in range(self.num_pass):
                    pass_idx = state.pass_idxes[i]
                    if self._in_taxi(pass_idx):
                        in_taxi.append(i)
                if len(in_taxi) > 0:
                    off = np.random.randint(len(in_taxi))
                    off_pass = in_taxi[off]
                    self._get_off_taxi(state.pass_idxes, off_pass, self.locs.index(taxi_loc))
            else:
                # 若没有人在车上却下车/出租车不在locs中记录的位置下车,则扣10分
                reward -= 10
        self._current_state = state
        self._total_reward += reward
        self._last_action = action
        return int(self.encode(state)), reward, self._done, {}

    def step(self, action: int) -> Tuple[int, int, bool, dict]:
        self._elapsed_steps += 1
        state, reward, done, info = self._step(action)
        if self._elapsed_steps >= self.max_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return state, reward, done, info

    def is_done(self):
        return self._done

    def print_state(self):
        print('current state:')
        for name, value in vars(self._current_state).items():
            print('%s=%s' % (name, value))

    def print(self):
        print("Taxi Env:")
        for name, value in vars(self).items():
            print('%s=%s' % (name, value))

    @staticmethod
    def _clear_shell():
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

    def render(self, show_info=True, clear=True):
        if clear:
            self._clear_shell()
        outfile = sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        state = self._current_state

        def ul(x):
            return "_" if x == " " else x

        has_pass = False
        for pass_idx in state.pass_idxes:
            if not self._in_taxi(pass_idx):
                pi, pj = self.locs[pass_idx]
                out[1 + pi][2 * pj + 1] = utils.colorize(
                    out[1 + pi][2 * pj + 1], "blue", bold=True
                )
            else:
                has_pass = True

        if not has_pass:
            out[1 + state.taxi_row][2 * state.taxi_col + 1] = utils.colorize(
                out[1 + state.taxi_row][2 * state.taxi_col + 1], "yellow", highlight=True
            )
        else:  # passenger in taxi
            out[1 + state.taxi_row][2 * state.taxi_col + 1] = utils.colorize(
                ul(out[1 + state.taxi_row][2 * state.taxi_col + 1]), "green", highlight=True
            )

        for dst_idx in state.dst_idxes:
            di, dj = self.locs[dst_idx]
            out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        if show_info:
            outfile.write(f'Taxi Loc: ({state.taxi_row},{state.taxi_col})\n')
            if self._last_action is not None:
                outfile.write(f"Last action: {self.EnumAction(self._last_action).name}")
            outfile.write("\n")
            outfile.write('Delivery list:\n')
            for i in range(self.num_pass):
                origin_idx = self._origin_state.pass_idxes[i]
                pass_idx = state.pass_idxes[i]
                dst_idx = state.dst_idxes[i]
                outfile.write(f'  Passenger {i}:  [{origin_idx}] {self.locs[origin_idx]}')
                outfile.write(f' -> [{dst_idx}] {self.locs[dst_idx]}  ')
                if self._in_taxi(pass_idx):
                    outfile.write('IN TAXI')
                elif pass_idx == dst_idx:
                    outfile.write('FINISH')
                else:
                    outfile.write('WAITING')
                outfile.write('\n')
            outfile.write("\n")
            outfile.write(f'TOTAL REWARD: {format(self._total_reward)}\n')
