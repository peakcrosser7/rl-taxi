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

    class EnumReward:
        MOVE = -1
        WRONG_OPT = -10
        RIGHT_PICK = 0
        RIGHT_DROP = 20
        # WRONG_OPT = -20
        # RIGHT_PICK = 10
        # RIGHT_DROP = 40

    class State:
        def __init__(self):
            self.taxi_row: int = 0
            self.taxi_col: int = 0
            self.pass_locs: List[int] = []
            self.dst_locs: List[int] = []

    def __init__(self, map_str: List[str],
                 locs_dict: Dict[str, Tuple[int, int]],
                 num_pass: int,
                 max_steps=200,
                 seed=None
                 ):
        self._desc = np.asarray(map_str, dtype="c")
        self.locs = []
        self._loc_map = []
        for w, (r, c) in locs_dict.items():
            self.locs.append((r, c))
            self._desc[r + 1, c * 2 + 1] = w[0]
            self._loc_map.append(w)
        locs = self.locs
        self._num_pass = num_pass
        self._max_steps = max_steps
        self._rng = np.random.RandomState(seed)

        self._num_rows = num_rows = len(map_str) - 2  # 去除最上最下两行
        self._num_cols = num_cols = int((len(map_str[0]) - 1) / 2)
        self._num_locs = num_locs = len(locs)

        self.n_action = self.EnumAction.__len__()
        self.n_observation = num_rows * num_cols * (num_locs + 1) ** num_pass * num_locs ** num_pass

        self._elapsed_steps = 0
        self._total_reward = 0
        self._done = False
        self._delivered = [False for _ in range(num_pass)]
        self._origin_state = self._init_state()
        # self._origin_state.taxi_col = self._origin_state.taxi_row = 0
        self._current_state = copy.deepcopy(self._origin_state)
        self._last_action = None

    def reset(self) -> State:
        self._elapsed_steps = 0
        self._total_reward = 0
        self._done = False
        self._delivered = [False for _ in range(self._num_pass)]
        self._origin_state = self._init_state()
        self._current_state = copy.deepcopy(self._origin_state)
        self._last_action = None
        return self.encode(self._current_state)

    def _init_state(self) -> State:
        loop = True
        while loop:
            i = self._rng.randint(self.n_observation)
            state = self.decode(i)
            loop = False
            # taxi_loc = (state.taxi_row, state.taxi_col)
            # if taxi_loc in self._locs:
            #     loop = True
            #     continue
            for i in range(self._num_pass):
                if state.pass_locs[i] == state.dst_locs[i] \
                        or self._in_taxi(state.pass_locs[i]):
                    loop = True
                    break
            if not loop:
                return state

    def seed(self, seed: int):
        self._rng = np.random.RandomState(seed)

    def encode(self, state: State):
        opt = state.taxi_row
        opt *= self._num_cols
        opt += state.taxi_col
        for i in range(self._num_pass):
            opt *= (self._num_locs + 1)
            opt += state.pass_locs[i]
        for i in range(self._num_pass):
            opt *= self._num_locs
            opt += state.dst_locs[i]
        return opt

    def decode(self, ipt: int):
        assert 0 <= ipt < self.n_observation
        state = self.State()
        dst_locs = []
        for i in range(self._num_pass):
            dst_locs.append(int(ipt % self._num_locs))
            ipt = ipt // self._num_locs
        dst_locs.reverse()
        state.dst_locs = dst_locs

        pass_locs = []
        for i in range(self._num_pass):
            pass_locs.append(int(ipt % (self._num_locs + 1)))
            ipt //= (self._num_locs + 1)
        pass_locs.reverse()
        state.pass_locs = pass_locs

        state.taxi_col = int(ipt % self._num_cols)
        ipt //= self._num_cols
        state.taxi_row = int(ipt)
        return state

    def _in_taxi(self, pass_loc: int) -> bool:
        return pass_loc == self._num_locs

    def _get_in_taxi(self, pass_locs: List[int], i: int):
        pass_locs[i] = self._num_locs

    @staticmethod
    def _get_off_taxi(pass_locs: List[int], i: int, new_loc: int):
        pass_locs[i] = new_loc

    def _is_delivered(self, pass_idx) -> bool:
        return self._delivered[pass_idx]

    def _num_delivered(self) -> int:
        """送达目的地的乘客数"""
        delivered = 0
        for d in self._delivered:
            delivered += d
        return delivered

    def _step(self, action: int) -> Tuple[int, int, bool, dict]:
        state = self._current_state
        reward = 0
        if self._done:
            return int(self.encode(state)), reward, self._done, {}

        taxi_loc = (row, col) = (state.taxi_row, state.taxi_col)
        act = self.EnumAction(action)
        if act == self.EnumAction.DOWN:
            state.taxi_row = min(row + 1, self._num_rows - 1)
            reward += self.EnumReward.MOVE
        elif act == self.EnumAction.UP:
            state.taxi_row = max(row - 1, 0)
            reward += self.EnumReward.MOVE
        elif act == self.EnumAction.RIGHT:
            if self._desc[1 + row, 2 * col + 2] == b":":
                state.taxi_col = min(col + 1, self._num_cols - 1)
            reward += self.EnumReward.MOVE
        elif act == self.EnumAction.LEFT:
            if self._desc[1 + row, 2 * col] == b":":
                state.taxi_col = max(col - 1, 0)
            reward += self.EnumReward.MOVE
        elif act == self.EnumAction.PICK_UP:
            match = False
            for i in range(self._num_pass):
                pass_loc = state.pass_locs[i]
                # 若乘客不在车上且出租车在乘客的位置
                if (not self._in_taxi(pass_loc)) and taxi_loc == self.locs[pass_loc] \
                        and (not self._is_delivered(i)):
                    self._get_in_taxi(state.pass_locs, i)
                    reward += self.EnumReward.RIGHT_PICK
                    match = True
                    break  # 一步只接一个人
            if not match:
                reward += self.EnumReward.WRONG_OPT
        elif act == self.EnumAction.DROP_OFF:
            match = False
            for i in range(self._num_pass):
                pass_loc = state.pass_locs[i]
                dst_loc = state.dst_locs[i]
                # 出租车在目的地且乘客在车上
                if taxi_loc == self.locs[dst_loc] and self._in_taxi(pass_loc):
                    self._get_off_taxi(state.pass_locs, i, dst_loc)
                    self._delivered[i] = True
                    reward += self.EnumReward.RIGHT_DROP
                    match = True
                    if self._num_delivered() == self._num_pass:
                        self._done = True
                    break
            # 在错误地点下车或出租车上没有人
            if not match:
                reward += self.EnumReward.WRONG_OPT
                # if taxi_loc in self.locs:
                #     in_taxi = []
                #     for i in range(self._num_pass):
                #         pass_loc = state.pass_locs[i]
                #         if self._in_taxi(pass_loc):
                #             in_taxi.append(i)
                #     if len(in_taxi) > 0:
                #         off = np.random.randint(len(in_taxi))
                #         off_pass = in_taxi[off]
                #         self._get_off_taxi(state.pass_locs, off_pass, self.locs.index(taxi_loc))
                # else:
                #     # 若没有人在车上却下车/出租车不在locs中记录的位置下车,则扣10分
                #     reward += self.EnumReward.WRONG_OPT

        self._current_state = state
        self._total_reward += reward
        self._last_action = action
        return int(self.encode(state)), reward, self._done, {}

    def step(self, action: int) -> Tuple[int, int, bool, dict]:
        self._elapsed_steps += 1
        state, reward, done, info = self._step(action)
        if self._elapsed_steps >= self._max_steps:
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

        out = self._desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        state = self._current_state

        def ul(x):
            return "_" if x == " " else x

        has_pass = False
        for i in range(self._num_pass):
            pass_loc = state.pass_locs[i]
            # 未上车且未到达目的地的乘客的位置标位蓝色
            if not self._in_taxi(pass_loc):
                if not self._is_delivered(i):
                    pi, pj = self.locs[pass_loc]
                    out[1 + pi][2 * pj + 1] = utils.colorize(
                        out[1 + pi][2 * pj + 1], "blue", bold=True
                    )
            else:
                has_pass = True

        # 未到达目的地的乘客的目的地标位红色
        for i in range(self._num_pass):
            dst_loc = state.dst_locs[i]
            if not self._is_delivered(i):
                di, dj = self.locs[dst_loc]
                out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "red")

        if not has_pass:  # 没有乘客的出租车为黄色
            out[1 + state.taxi_row][2 * state.taxi_col + 1] = utils.colorize(
                out[1 + state.taxi_row][2 * state.taxi_col + 1], "yellow", highlight=True
            )
        else:  # 有出租车的乘客为绿色
            out[1 + state.taxi_row][2 * state.taxi_col + 1] = utils.colorize(
                ul(out[1 + state.taxi_row][2 * state.taxi_col + 1]), "green", highlight=True
            )

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        # self.print_state()
        if show_info:
            outfile.write(f'Taxi Loc: ({state.taxi_row},{state.taxi_col})\n')
            if self._last_action is not None:
                outfile.write(f"Last action: {self.EnumAction(self._last_action).name}")
            outfile.write("\n")
            outfile.write('Delivery list:\n')
            for i in range(self._num_pass):
                origin_loc = self._origin_state.pass_locs[i]
                pass_loc = state.pass_locs[i]
                dst_loc = state.dst_locs[i]
                outfile.write(f'  Passenger {i}:  [{self._loc_map[origin_loc]}] {self.locs[origin_loc]}')
                outfile.write(f' -> [{self._loc_map[dst_loc]}] {self.locs[dst_loc]}  ')
                if self._in_taxi(pass_loc):
                    outfile.write('IN TAXI')
                elif pass_loc == dst_loc:
                    outfile.write('FINISH')
                else:
                    outfile.write('WAITING')
                outfile.write('\n')
            outfile.write("\n")
            if self.is_done():
                outfile.write('GAME DONE\n')
            outfile.write(f'TOTAL REWARD: {format(self._total_reward)}\n')
