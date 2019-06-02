import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from common.board import Board
from common.gamestate import GameState
from common.move import Move
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

    def add_child_node(self, point, child_node):
        self._children_node[point] = child_node

    def has_child_node(self, point):
        return point in self._children_node

    def get_child_node(self, point):
        return self._children_node[point]

    def expected_value_of_branch(self, point):
        branch = self._children_branch[point]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value/branch.visit_count

    def prior_of_branch(self, point):
        return self._children_branch[point].prior

    def visit_count_of_branch(self, point):
        if point in self._children_branch:
            return self._children_branch[point].visit_count
        return 0

    def record_visit(self, point, value):
        self._total_visit_count += 1
        self._children_branch[point].visit_count += 1
        self._children_branch[point].total_value += value

    def select_branch(self):
        def score_branch(point):
            q = self.expected_value_of_branch(point)
            p = self.prior_of_branch(point)
            n = self.visit_count_of_branch(point)
            return q + Node.temperature*p*np.sqrt(self._total_visit_count)/(n+1)
        return max(self.children_branch(), key=score_branch)

    def visit_count(self, point):
        if point in self._children_branch:
            return self._children_branch[point].visit_count
        return 0


class AlphaZeroExperienceCollector:
    def __init__(self):
        self._states = []
        self._visit_counts = []
        self._rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def _reset_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._reset_episode()

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self._states += self._current_episode_states
        self._visit_counts += self._current_episode_visit_counts
        self._rewards += [reward for _ in range(num_states)]

        self._reset_episode()

    @property
    def visit_count(self):
        return self._visit_counts

    @property
    def rewards(self):
        return self._rewards


class AlphaZeroAgent(Player):
    def __init__(self, encoder, model, num_rounds, experience_collector):
        self._encoder = encoder
        self._model = model
        self._num_rounds = num_rounds
        self._experience_collector = experience_collector
        self._criterion_policy = nn.CrossEntropyLoss()
        self._criterion_value = nn.MSELoss()
        self._optimizer = optim.SGD(self._model.parameters(),lr=0.002)


    def create_node(self, game_state, parent_branch=None, parent_node=None):
        board_matrix = self._encoder.encode(game_state)
        estimated_branch_priors, estimated_state_value = self._model(
            board_matrix)
        chiildren_branch = {}
        for idx, p in enumerate(estimated_branch_priors):
            point = self._encoder.decode_point_index(idx)
            if game_state.board.is_free_point(point):
                chiildren_branch[point] = Branch(Move(point), p)

        new_node = Node(game_state, estimated_state_value,
                        chiildren_branch, parent_branch, parent_node)
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

        board_matrix = self._encoder.encode(game_state)
        visit_counts = np.array([root.visit_count(self._encoder.decode_point_index(
            idx)) for idx in range(self._encoder.num_points())])
        self._experience_collector.record_decision(board_matrix, visit_counts)

        return max(root.children_branch(), key=root.visit_count)

    def train(self, expeience, learning_rate, batch_size):
        
        self._model.train()
        
        num_examples = expeience.states.shape[0]
        model_input = expeience.states

        visit_sums = np.sum(expeience.visit_counts,axis=1).reshape((num_examples, 1))
        action_policy_target = expeience.visit_counts / visit_sums

        value_target = expeience.rewards
        [action_policy, value] = self._model(model_input)

        loss_policy = self._criterion_policy(action_policy, action_policy_target)
        loss_value = self._criterion_value(value, value_target)

        self._optimizer.zero_grad()
        loss= loss_policy + loss_value
        loss.backward()
        self._optimizer.step()
