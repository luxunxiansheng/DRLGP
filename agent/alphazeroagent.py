import copy

import math
import random

from tqdm import tqdm

from common.gamestate import GameState
from common.move import Move
from common.board import Board
from common.player import Player


class Branch:
    def __init__(self, move, prior):
        self._move = move
        self._prior = prior
        self._taken_count = 0
        self._total_value = 0.0

    @property
    def move(self):
        return self._move

    @property
    def prior(self):
        return self._prior

    @property
    def total_value(self):
        return self._total_value

    @property
    def taken_count(self):
        return self._taken_count


class Node:
    def __init__(self, game_state, game_state_value, priors, parent_node):
        self._game_state = game_state
        self._game_state_value = game_state_value
        self._parent_node = parent_node
        self._total_visit_count = 1
        self._branches = {}
        for move, prior in priors.items():
            if game_state.board.is_free_point(move.point):
                self._branches[move] = Branch(move, prior)

        self._child_nodes = {}

    @property
    def branches(self):
        return self._branches

    def add_child(self, branch, child_node):
        self._child_nodes[branch] = child_node

    def has_child(self, branch):
        return branch in self._child_nodes

    def expected_value(self, move):
        branch = self._branches[move]
        if branch.taken_count == 0:
            return 0.0
        return branch.total_value/branch.taken_count

    def prior(self, move):
        return self.branches[move].prior

    def taken_count(self, move):
        if move in self.branches:
            return self.branches[move].taken_count
        return 0
    
    def score_branch(self,branch):
        q_value= self.expected_value()



    def select_branch(self):
        



class AlphaZeroAgent(Player):
    
    def select_move(self,game,game_state):

    