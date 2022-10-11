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
    class Actions(Enum):
        UP = 0,
        DOWN = 1,
        LEFT = 2,
        RIGHT = 3,
        PICK_UP = 4,
        DROP_OFF = 5

    def __init__(self, map_str: List[str], locs: List[Tuple[int, int]],
                 num_pass: int):
        self.desc = np.asarray(map_str, dtype="c")
        self.locs = locs
        self.num_pass = num_pass

        self.num_rows = num_rows = len(map_str) - 2  # 去除最上最下两行
        self.num_cols = num_cols = (len(map_str[0]) - 1) / 2
        self.num_locs = num_locs = len(locs)

        self.action_space = self.Actions.__len__()

        self.observation_space = num_rows * num_cols * (num_locs + 1) ** num_pass * num_locs ** num_pass
        self.np_random, seed = seeding.np_random()

    def encode(self, taxi_row: int, taxi_col: int, pass_locs: List[int], dst_locs: List[int]):
        opt = taxi_row
        opt *= self.num_rows
        opt += taxi_col
        opt *= self.num_cols
        for i in range(self.num_pass):
            opt += pass_locs[i]
            opt *= (self.num_locs + 1)
        for i in range(self.num_pass - 1):
            opt += dst_locs[i]
            opt *= self.num_locs
        opt += dst_locs[self.num_pass - 1]
        return opt

    def decode(self, ipt):
        assert 0 <= ipt < self.observation_space
        out = []
        dst_locs = []
        for i in range(self.num_pass - 1):
            dst_locs.append(ipt % self.num_locs)
            ipt = ipt // self.num_locs
        dst_locs.append(ipt % (self.num_locs + 1))
        ipt //= self.num_locs + 1
        dst_locs.reverse()
        out.append(dst_locs)

        pass_locs = []
        for i in range(self.num_pass - 1):
            pass_locs.append(ipt % (self.num_locs + 1))
            ipt //= (self.num_locs + 1)
        pass_locs.append(ipt % self.num_cols)
        ipt //= self.num_cols
        pass_locs.reverse()
        out.append(pass_locs)

        out.append(ipt % self.num_rows)
        ipt //= self.num_rows
        out.append(ipt)
        return reversed(out)
