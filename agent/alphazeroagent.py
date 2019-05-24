import copy

import math
import random
import numpy as np

from tqdm import tqdm

from common.gamestate import GameState
from common.move import Move
from common.board import Board
from common.player import Player


class Branch:
    def __init__(self, move, prior):
        self._move = move
        self._prior = prior
        self._visit_count = 0
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
    def visit_count(self):
        return self._visit_count


class Node:

    temperature = 0.5

    def __init__(self, game_state, game_state_value, children_branch, parent_node, parent_branch):
        self._game_state = game_state
        self._game_state_value = game_state_value
        self._parent_node = parent_node
        self._parent_branch = parent_branch
        self._total_visit_count = 1
        self._children_branch = children_branch
        self._children_node = {}

    @property
    def game_state(self):
        return self._game_state

    @property
    def game_state_value(self):
        return self._game_state_value

    def children_branch(self):
        return self._children_branch.keys()

    def add_child_node(self, branch, child_node):
        self._children_node[branch] = child_node

    def has_child_node(self, branch):
        return branch in self._children_node

    def get_child_node(self, branch):
        return self._children_node[branch]

    def expected_value_of_branch(self, branch):
        branch = self._children_branch[branch]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value/branch.visit_count

    def prior_of_branch(self, branch):
        return self._children_branch[branch].prior

    def visit_count_of_branch(self, branch):
        if branch in self._children_branch:
            return self._children_branch[branch].visit_count
        return 0

    def record_visit(self, branch, value):
        self._total_visit_count += 1
        self._children_branch[branch].visit_count += 1
        self._children_branch[branch].total_value += value

    def select_branch(self):
        def score_branch(branch):
            q = self.expected_value_of_branch(branch)
            p = self.prior_of_branch(branch)
            n = self.visit_count_of_branch(branch)
            return q + Node.temperature*p*np.sqrt(self._total_visit_count)/(n+1)
        return max(self.children_branch(), key=score_branch)

    def visit_count(self, branch):
        if branch in self._children_branch:
            return self._children_branch[branch].visit_count
        return 0


class AlphaZeroAgent(Player):
    def __init__(self, encoder, model, num_rounds):
        self._encoder = encoder
        self._model = model
        self._num_rounds = num_rounds

    def create_node(self, game_state, parent_branch=None, parent_node=None):
        state_tensor = self._encoder.encode(game_state)
        estimated_branch_priors, estimated_state_value = self._model(
            state_tensor)
        chiildren_branch = {}
        for idx, p in enumerate(estimated_branch_priors):
            point = self._encoder.decode_move_index(idx)
            if game_state.board.is_free_point(point):
                chiildren_branch[Branch(Move(point), p)] = p

        new_node = Node(game_state, estimated_state_value,chiildren_branch, parent_branch, parent_node)
        if parent_node is not None:
            parent_node.add_child_node(parent_branch, new_node)
        return new_node

    def select_move(self, game, game_state):
        root = self.create_node(game_state)
        for _ in range(self._num_rounds):
            node = root
            
            # select
            next_branch = node.select_branch()
            while node.has_child_node(next_branch):
                node = node.get_child_node(next_branch)
                next_branch = node.select_branch()

            # expand 
            new_state = game.transit(node.game_state, next_branch.move)
            parent_branch = next_branch
            parent_node = node
            new_node = self.create_node(new_state, parent_branch, parent_node)

            # backup 
            value = -1*new_node.game_state_value
            while parent_node is not None:
                parent_node.record_visit(parent_branch, value)
                parent_branch = parent_node.parent_branch
                parent_node = parent_node.parent_node
                value = -1 * value

        return max(root.children_branch(), key=root.visit_count)
