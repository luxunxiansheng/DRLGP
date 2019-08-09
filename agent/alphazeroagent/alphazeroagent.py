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

from common.board import Board
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from .mcts.branch import Branch
from .mcts.node import Node
from .mcts.tree import Tree


class AlphaZeroAgent(Player):
    def __init__(self, id, name, mark, encoder, model, num_rounds, temperature, experience_collector=None, device='cpu'):
        super().__init__(id, name, mark)
        self._encoder = encoder
        self._device = device
        self._model = model
        self._num_rounds = num_rounds
        self._experience_collector = experience_collector
        self._temperature = temperature
        self._mcts_tree   = Tree()

    def reset(self):
        #TODO:  reset tree?
       
        if self._experience_collector is not None:
            self._experience_collector.reset_episode()

    @property
    def experience_collector(self):
        return self._experience_collector

    @property
    def temperature(self):
        return self._temperature

    def set_experience_collector(self, experience_collector):
        self._experience_collector = experience_collector

    def create_node(self, game_state, estimated_branch_priors, estimated_state_value, parent_branch=None):

        new_node = Node(game_state, estimated_state_value, parent_branch, self._temperature)

        chiildren_branch = {}
        for idx, p in enumerate(estimated_branch_priors):
            point = self._encoder.decode_point_index(idx)
            if game_state.board.is_free_point(point):
                chiildren_branch[point] = Branch(new_node, Move(point), p)

        new_node.children_branch = chiildren_branch

        if parent_branch is not None:
            parent_branch.child_node = new_node

        return new_node

    def select_move(self, game, game_state):  # it is guaranteed that it is not the final game state anyway
        game.state_cache.push(game_state.board)
        root_board_matrix = self._encoder.encode(game.state_cache.game_states)
        model_input = torch.from_numpy(root_board_matrix).unsqueeze(0).to(self._device, dtype=torch.float)
        estimated_branch_priors, estimated_state_value = self.predict(model_input)
        root = self.create_node(game_state, estimated_branch_priors[0], estimated_state_value[0].item())

        for _ in tqdm(range(self._num_rounds)):
            node = root
            game_state_memory = copy.deepcopy(game.state_cache)

            # select
            next_branch = node.select_branch(is_root=True, is_selfplay=game.is_selfplay)
            assert next_branch is not None

            # search the tree until the game end node or a new node
            while next_branch is not None:
                if next_branch.child_node is not None:
                    node = next_branch.child_node
                    game_state_memory.push(node.game_state.board)
                    next_branch = node.select_branch(is_root=False, is_selfplay=game.is_selfplay)

                else:
                    # expand
                    new_state = game.look_ahead_next_move(node.game_state, next_branch.move)
                    game_state_memory.push(node.game_state.board)
                    temp_board_matrix = self._encoder.encode(game_state_memory.game_states)
                    model_input = torch.from_numpy(temp_board_matrix).unsqueeze(0).to(self._device, dtype=torch.float)
                    estimated_branch_priors, estimated_state_value = self.predict(model_input)
                    node = self.create_node(new_state, estimated_branch_priors[0], estimated_state_value[0].item(), next_branch)
                    break

            value = -1 * node.game_state_value

            parent_branch = node.parent_branch
            while parent_branch is not None:
                parent_node = parent_branch.parent_node
                parent_node.record_visit(parent_branch.move.point, value)
                parent_branch = parent_node.parent_branch
                value = -1 * value

        visit_counts = np.array([root.visit_counts_of_branch(self._encoder.decode_point_index(idx)) for idx in range(self._encoder.num_points())])

        if self._experience_collector is not None:
            self._experience_collector.record_decision(root_board_matrix, visit_counts)

        return Move(max(root.children_branch, key=root.visit_counts_of_branch))

    def predict(self, input_states):
        return self._model(input_states)
