# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /


import copy
import gc
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from agent.alphazeroagent.experiencecollector import ExperienceCollector

from common.board import Board
from common.gamestate import GameState
from common.move import Move
from common.player import Player


class Branch(object):
    def __init__(self, parent_node,move,prior):
        self._move = move
        self._total_value = 0.0
        self._prior = prior
        self._visit_counts = 0

        self._parent_node  = parent_node
        self._child_node = None
    
    @property
    def move(self):
        return self._move

    @property
    def prior(self):
        return self._prior

    @property
    def total_value(self):
        return self._total_value

    @total_value.setter
    def total_value(self, value):
        self._total_value = value

    @property
    def visit_counts(self):
        return self._visit_counts

    @visit_counts.setter
    def visit_counts(self, value):
        self._visit_counts = value
    
    @property
    def expected_value(self):
        return self._total_value/self._visit_counts if self._visit_counts!=0 else 0

    @property
    def parent_node(self):
        return self._parent_node

    @property
    def child_node(self):
        return self._child_node    

    def add_child_node(self, node):
        self._child_node = node

class Node(object):
    def __init__(self, game_state, game_state_value, parent_branch, c_puct=5.0):
        self._game_state = game_state
        
        self._game_state_value = game_state_value
        self._total_visit_counts = 1

        self._parent_branch = parent_branch
        self._children_branch = {}

        self._c_puct = c_puct


    def add_branch(self,point,prior):

        self._children_branch[point]=Branch(self,Move(point),prior)

    def does_branch_exist(self,point):
        return point in self.children_branch

    def get_child_branch(self, point):
        return self._children_branch[point]
        
    def expected_value_of_branch(self, point):
        return self._children_branch[point].expected_value
        
    def prior_of_branch(self, point):
        return self._children_branch[point].prior
        
    def visit_counts_of_branch(self, point):
        return self._children_branch[point].visit_counts if self.does_branch_exist(point) else 0
        
        
    def record_visit(self, point, value):
        self._total_visit_counts += 1
        self._children_branch[point].visit_counts += 1
        self._children_branch[point].total_value += value

    def select_branch(self):
        Qs = [self.expected_value_of_branch(point) for point in self.children_branch]
        Ps = [self.prior_of_branch(point) for point in self.children_branch]
        Ns = [self.visit_counts_of_branch(point) for point in self.children_branch]
        
        scores = [(q + self._c_puct * p * np.sqrt(self._total_visit_counts) / (n + 1)).item() for q, p, n in zip(Qs, Ps, Ns)]
        best_point_index = np.argmax(scores)

        points = list(self.children_branch)
        return self._children_branch[points[best_point_index]]

    @property
    def game_state(self):
        return self._game_state

    @property
    def parent_branch(self):
        return self._parent_branch

    @property
    def game_state_value(self):
        return self._game_state_value

    @property
    def temperature(self):
        return self._c_puct

    @property
    def children_branch(self):
        return self._children_branch.keys()

    @children_branch.setter
    def children_branch(self, value):
        self._children_branch = value

    def is_leaf(self):
        return not self._children_branch 


class AlphaZeroTree(object):
    def __init__(self):
        self._working_node = None
    
    @property
    def working_node(self):    
        return self._working_node

    @working_node.setter
    def working_node(self, value):
        self._working_node = value

    def reset(self):
        self._working_node =None    
    
    
    def go_down(self, move):
        if self._working_node is not None:
            branch = self._working_node.get_child_branch(move.point)
            self._working_node = branch.child_node


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class AlphaZeroAgent(Player):
    def __init__(self, id, name, encoder, model, mcts_tree, num_rounds, c_puct, temperature=1.0, device='cpu'):
        super().__init__(id, name)
        self._encoder = encoder
        self._device = device
        self._model = model.to(device)
        self._num_rounds = num_rounds
        self._experience_collector = ExperienceCollector()
        self._mcts_tree = mcts_tree
        self._cpuct = c_puct
        self._temperature = temperature
    
    @property
    def msct_tree(self):
        return self._mcts_tree

    @property
    def experience_collector(self):
        return self._experience_collector

    def _predict(self, board_matrix):
        model_input = torch.from_numpy(board_matrix).unsqueeze(0).to(self._device, dtype=torch.float)
        return self._model(model_input)

    def _create_node(self, game_state, estimated_state_value, estimated_branch_priors, parent_branch):

        new_node = Node(game_state, estimated_state_value,parent_branch, self._cpuct)

        if estimated_branch_priors is not None:
            for idx, p in enumerate(estimated_branch_priors):
                point = self._encoder.decode_point_index(idx)
                if new_node.game_state.board.is_free_point(point):
                    new_node.add_branch(point, p)
        return new_node
        

    def select_move(self, game):
        # encode  the last specified boards as the root
        root_board_matrix = self._encoder.encode(
            game.state_cache.game_states, game.working_game_state.player_in_action, game.working_game_state.previous_move)

        if self._mcts_tree.working_node is None:
            estimated_branch_priors, estimated_state_value = self._predict(
                root_board_matrix)
            self._mcts_tree.working_node = self._create_node(
                game.working_game_state, estimated_state_value[0].item(), estimated_branch_priors[0], None)

        working_root = self._mcts_tree.working_node

        for _ in tqdm(range(self._num_rounds), desc='Rollout Loop'):
            current_node = working_root
            game_state_memory = copy.deepcopy(game.state_cache)

            while True:
                # reach  the end of the game
                if game.is_final_state(current_node.game_state):
                    break

                current_branch = current_node.select_branch()
                if current_branch.child_node is None:
                    # expand
                    new_state = game.look_ahead_next_move(
                        current_node.game_state, current_branch.move)
                    game_state_memory.push(new_state)

                    value_of_new_state = 0
                    priors_of_new_children_branch = None

                    #  cope with the end state
                    if game.is_final_state(new_state):
                        winner = game.get_winner(new_state)
                        if winner is not None:
                            if winner.id == self.id:
                                value_of_new_state = -1
                            else:
                                value_of_new_state = 1
                        else:
                            value_of_new_state = 0
                    #  the normal state
                    else:
                        board_matrix = self._encoder.encode(game_state_memory.game_states, new_state.player_in_action,new_state.previous_move)
                        estimated_branch_priors, estimated_state_value = self._predict(
                            board_matrix)
                        value_of_new_state = estimated_state_value[0].item()
                        priors_of_new_children_branch = estimated_branch_priors[0]

                    current_node = self._create_node(new_state, value_of_new_state, priors_of_new_children_branch, current_branch)
                    current_branch.add_child_node(current_node)
                    break
                else:
                    current_node = current_branch.child_node
                    game_state_memory.push(current_node.game_state)

            # Normally , the parent node value sholud be the oppsite of current node's value
            value = -1 * current_node.game_state_value

            while True:
                current_branch = current_node.parent_branch
                current_node = current_branch.parent_node
                current_node.record_visit(current_branch.move.point, value)
                value = -1 * value
                if current_node == working_root:
                    break

        free_points = []
        visit_counts_of_free_points = []
        visit_counts = []

        for idx in range(self._encoder.num_points()):
            point = self._encoder.decode_point_index(idx)
            visit_count = working_root.visit_counts_of_branch(point)
            visit_counts.append(visit_count)
            if working_root.does_branch_exist(point):
                free_points.append(point)
                visit_counts_of_free_points.append(visit_count)

        if game.is_selfplay:
            next_move_probabilities = softmax(np.log(np.asarray(visit_counts_of_free_points)+1e-10))

            # add dirichlet noise for exploration
            next_move_probabilities = 0.75 * next_move_probabilities + 0.25 * \
                np.random.dirichlet(
                    0.3 * np.ones(len(next_move_probabilities)))
            next_move = Move(free_points[np.random.choice(
                len(free_points), p=next_move_probabilities)])
            self._experience_collector.record_decision(
                root_board_matrix, np.asarray(visit_counts))
        else:
            next_move_probabilities = softmax(
                1.0/self._temperature*np.log(np.asarray(visit_counts_of_free_points)+1e-10))

            next_move = Move(free_points[np.random.choice(
                len(free_points), p=next_move_probabilities)])

        self._mcts_tree.go_down(next_move)            
        return next_move
