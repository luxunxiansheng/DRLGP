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
from agent.alphazeroagent.mcts.branch import Branch
from agent.alphazeroagent.mcts.node import Node
from agent.alphazeroagent.mcts.tree import Tree
from common.board import Board
from common.gamestate import GameState
from common.move import Move
from common.player import Player


class AlphaZeroAgent(Player):
    def __init__(self, id, name, mark, encoder, model, mcts_tree, num_rounds, device='cpu'):
        super().__init__(id, name, mark)
        self._encoder = encoder
        self._device = device
        self._model = model
        self._num_rounds = num_rounds
        self._experience_collector = ExperienceCollector()
        self._mcts_tree = mcts_tree

    @property
    def experience_collector(self):
        return self._experience_collector

    def _predict(self, board_matrix):
        model_input = torch.from_numpy(board_matrix).unsqueeze(0).to(self._device, dtype=torch.float)
        return self._model(model_input)

    def _create_node_with_children_branch(self, game_state, estimated_state_value, estimated_branch_priors, parent_branch):

        new_node = Node(game_state, estimated_state_value, parent_branch)

        for idx, p in enumerate(estimated_branch_priors):
            point = self._encoder.decode_point_index(idx)
            if new_node.game_state.board.is_free_point(point):
                new_node.add_branch(point, p)
        return new_node

    @profile
    def select_move(self, game):
        root_board_matrix = self._encoder.encode(game.state_cache.game_states)

        if self._mcts_tree.working_node is None:
            estimated_branch_priors, estimated_state_value = self._predict(root_board_matrix)
            self._mcts_tree.working_node = self._create_node_with_children_branch(game.working_game_state, estimated_state_value[0].item(), estimated_branch_priors[0], None)

        working_root = self._mcts_tree.working_node

        for _ in tqdm(range(self._num_rounds)):
            current_node = working_root
            game_state_memory = copy.deepcopy(game.state_cache)

            while True:
                # reach  the end of the game
                if current_node.is_leaf():
                    break

                randomly = True if current_node == working_root else False
                current_branch = current_node.select_branch(randomly)
                if current_branch.child_node is None:
                    # expand
                    new_state = game.look_ahead_next_move(current_node.game_state, current_branch.move)
                    game_state_memory.push(new_state)
                    board_matrix = self._encoder.encode(game_state_memory.game_states)
                    estimated_branch_priors, estimated_state_value = self._predict(board_matrix)
                    current_node = self._create_node_with_children_branch(new_state, estimated_state_value[0].item(), estimated_branch_priors[0], current_branch)
                    current_branch.add_child_node(current_node)
                    break
                else:
                    current_node = current_branch.child_node
                    game_state_memory.push(current_node.game_state)

            # update
            value = -1 * current_node.game_state_value
            while True:
                current_branch = current_node.parent_branch
                current_node = current_branch.parent_node
                current_node.record_visit(current_branch.move.point, value)
                value = -1 * value
                if current_node == working_root:
                    break

        visit_counts = np.array([working_root.visit_counts_of_branch(self._encoder.decode_point_index(idx)) for idx in range(self._encoder.num_points())])
        next_move = Move(max(working_root.children_branch, key=working_root.visit_counts_of_branch))

        if game.is_selfplay:
            self._experience_collector.record_decision(root_board_matrix, visit_counts)
            self._mcts_tree.go_down(next_move)
        else:
            self._mcts_tree.working_node = None

        return next_move
