import math
import sys
from contextlib import closing
from enum import Enum
from io import StringIO
from typing import List, Tuple

from gym import utils
import numpy as np
from gym.utils import seeding


class TaxiEnv:
    class EnumAction(Enum):
        DOWN = 0,
        UP = 1,
        RIGHT = 2,
        LEFT = 3,
        PICK_UP = 4,
        DROP_OFF = 5

    class State:
        def __init__(self):
            self.taxi_row: int = 0
            self.taxi_col: int = 0
            self.pass_idxes: List[int] = []
            self.dst_idxes: List[int] = []

    def __init__(self, map_str: List[str], locs: List[Tuple[int, int]],
                 num_pass: int, max_steps=200):
        self.desc = np.asarray(map_str, dtype="c")
        self.locs = locs
        self.num_pass = num_pass
        self.max_steps = max_steps

        self.num_rows = num_rows = len(map_str) - 2  # 去除最上最下两行
        self.num_cols = num_cols = (len(map_str[0]) - 1) / 2
        self.num_locs = num_locs = len(locs)

        self.action_space = self.EnumAction.__len__()

        self.observation_space = num_rows * num_cols * (num_locs + 1) ** num_pass * num_locs ** num_pass
        self.np_random, seed = seeding.np_random()

        self._elapsed_steps = 0
        self._total_reward = 0
        self._current_state = self._init_state()
        self._left_pass = num_pass
        self._last_action = None

    def _init_state(self) -> State:
        return self.State()

    def encode(self, state: State):
        opt = state.taxi_row
        opt *= self.num_rows
        opt += state.taxi_col
        opt *= self.num_cols
        for i in range(self.num_pass):
            opt += state.pass_idxes[i]
            opt *= (self.num_locs + 1)
        for i in range(self.num_pass - 1):
            opt += state.dst_idxes[i]
            opt *= self.num_locs
        opt += state.dst_idxes[self.num_pass - 1]
        return opt

    # def encode(self, taxi_row: int, taxi_col: int, pass_locs: List[int], dst_locs: List[int]):
    #     opt = taxi_row
    #     opt *= self.num_rows
    #     opt += taxi_col
    #     opt *= self.num_cols
    #     for i in range(self.num_pass):
    #         opt += pass_locs[i]
    #         opt *= (self.num_locs + 1)
    #     for i in range(self.num_pass - 1):
    #         opt += dst_locs[i]
    #         opt *= self.num_locs
    #     opt += dst_locs[self.num_pass - 1]
    #     return opt

    def decode(self, ipt: int):
        assert 0 <= ipt < self.observation_space
        state = self.State()
        dst_locs = []
        for i in range(self.num_pass - 1):
            dst_locs.append(ipt % self.num_locs)
            ipt = ipt // self.num_locs
        dst_locs.append(ipt % (self.num_locs + 1))
        ipt //= self.num_locs + 1
        dst_locs.reverse()
        state.dst_idxes = dst_locs

        pass_locs = []
        for i in range(self.num_pass - 1):
            pass_locs.append(ipt % (self.num_locs + 1))
            ipt //= (self.num_locs + 1)
        pass_locs.append(ipt % self.num_cols)
        ipt //= self.num_cols
        pass_locs.reverse()
        state.pass_idxes = pass_locs

        state.taxi_col = ipt % self.num_rows
        ipt //= self.num_rows
        state.taxi_row = ipt
        return state

    # def decode(self, ipt):
    #     assert 0 <= ipt < self.observation_space
    #     out = []
    #     dst_locs = []
    #     for i in range(self.num_pass - 1):
    #         dst_locs.append(ipt % self.num_locs)
    #         ipt = ipt // self.num_locs
    #     dst_locs.append(ipt % (self.num_locs + 1))
    #     ipt //= self.num_locs + 1
    #     dst_locs.reverse()
    #     out.append(dst_locs)
    #
    #     pass_locs = []
    #     for i in range(self.num_pass - 1):
    #         pass_locs.append(ipt % (self.num_locs + 1))
    #         ipt //= (self.num_locs + 1)
    #     pass_locs.append(ipt % self.num_cols)
    #     ipt //= self.num_cols
    #     pass_locs.reverse()
    #     out.append(pass_locs)
    #
    #     out.append(ipt % self.num_rows)
    #     ipt //= self.num_rows
    #     out.append(ipt)
    #     return reversed(out)

    def _in_taxi(self, pass_idx: int):
        return pass_idx == self.num_locs

    def _get_in_taxi(self, pass_idxes: List[int], i: int):
        pass_idxes[i] = self.num_pass

    def _get_off_taxi(self, pass_idxes: List[int], i: int, new_loc_idx: int):
        pass_idxes[i] = new_loc_idx

    def _step(self, action: int) -> Tuple[int, int, bool, dict]:
        act = self.EnumAction(action)
        reward = -1
        done = False
        state = self._current_state
        taxi_loc = (row, col) = (state.taxi_row, state.taxi_col)

        if act == self.EnumAction.DOWN:
            state.taxi_row = min(row, self.num_rows - 1)
        elif act == self.EnumAction.UP:
            state.taxi_row = max(row, 0)
        elif act == self.EnumAction.RIGHT and self.desc[1 + row, 2 * col + 2] == b":":
            state.taxi_col = min(col + 1, self.num_cols - 1)
        elif act == self.EnumAction.LEFT and self.desc[1 + row, 2 * col] == b":":
            state.taxi_col = min(col - 1, 0)
        elif act == self.EnumAction.PICK_UP:
            match = False
            for i in range(self.num_pass):
                pass_idx = state.pass_idxes[i]
                if (not self._in_taxi(pass_idx)) and taxi_loc == self.locs[pass_idx]:
                    self._get_in_taxi(state.pass_idxes, i)
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
                    reward += 20
                    match = True
                    self._left_pass -= 1
                    if self._left_pass == 0:
                        done = True
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
        return int(self.encode(state)), reward, done, {}

    def step(self, action: int) -> Tuple[int, int, bool, dict]:
        self._elapsed_steps += 1
        state, reward, done, info = self._step(action)
        if self._elapsed_steps >= self.max_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return state, reward, done, info

    def render(self):
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
        if self._last_action is not None:
            outfile.write("  Last Action: {}".format(
                self.EnumAction(self._last_action).name
            )
            )
        outfile.write("\n")

        if has_pass:
            in_taxi = []
            for i in range(self.num_pass):
                if self._in_taxi(state.pass_idxes[i]):
                    in_taxi.append(i)
            outfile.write("  Pass in taxi: {}".format(in_taxi))
        outfile.write("\n")
