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

from agent.experiencecollector import ExperienceCollector
from common.board import Board
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from agent.mcts.mctsnode import MCTSNode
from agent.mcts.mctstree import MCTSTree


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class AlphaZeroAgent(Player):
    def __init__(self, id, name, encoder, model, num_rounds, c_puct, temperature=1.0, device='cpu'):
        super().__init__(id, name)
        self._encoder = encoder
        self._device = device
        self._model = model.to(device)
        self._num_rounds = num_rounds
        self._experience_collector = ExperienceCollector()
        self._mcts_tree = MCTSTree()
        self._cpuct = c_puct
        self._temperature = temperature
    
    @property
    def mcts_tree(self):
        return self._mcts_tree

    @mcts_tree.setter
    def mcts_tree(self,tree):
        self._mcts_tree = tree


    @property
    def experience_collector(self):
        return self._experience_collector

    def _predict(self, board_matrix):
        model_input = torch.from_numpy(board_matrix).unsqueeze(0).to(self._device, dtype=torch.float)
        return self._model(model_input)


    def select_move(self, game):
        
        # encode  the last specified boards as the root
        root_board_matrix = self._encoder.encode(game.state_cache.game_states, game.working_game_state.player_in_action, game.working_game_state.previous_move)

        if self._mcts_tree.working_node is None:
            self._mcts_tree.working_node = MCTSNode(game.working_game_state,1.0,None)

        for _ in tqdm(range(self._num_rounds), desc='Rollout Loop'):
            node = self._mcts_tree.working_node
            game_state_memory = copy.deepcopy(game.state_cache)

            while True:
                if node.is_leaf():
                    break
                node = node.select(self._cpuct)
                game_state_memory.push(node.game_state)
                

            leaf_value = 0.0 
            if not game.is_final_state(node.game_state):
                # encode  the last specified boards as the root
                board_matrix = self._encoder.encode(game_state_memory.game_states, node.game_state.player_in_action, node.game_state.previous_move)
                estimated_priors , estimated_state_value = self._predict(board_matrix) 
                leaf_value = estimated_state_value[0].item()
                
                free_points=node.game_state.board.get_legal_points()
                for point in free_points:
                    idx = self._encoder.encode_point(point)
                    node.add_child(game,point,estimated_priors[0][idx].item())
            else:
                if game.final_winner is not None:
                    leaf_value = 1.0 if game.final_winner == node.game_state.player_in_action else -1.0

            node.update_recursively(self._mcts_tree.working_node,-leaf_value)

        #self._mcts_tree.working_node.game_state.board.print_visits(self._mcts_tree.working_node.children)   
                    
        free_points = []
        visit_counts_of_free_points = []
        visit_counts = []

        for idx in range(self._encoder.num_points()):
            point = self._encoder.decode_point_index(idx)
            
            num_visits= 0
            if point in self.mcts_tree.working_node.children:
                free_points.append(point)
                num_visits = self.mcts_tree.working_node.get_child(point).num_visits
                visit_counts_of_free_points.append(num_visits)
            
            visit_counts.append(num_visits)
            
        if game.is_selfplay:
            next_move_probabilities = softmax(np.log(np.asarray(visit_counts_of_free_points)+1e-10))

            # add dirichlet noise for exploration
            next_move_probabilities = 0.75 * next_move_probabilities + 0.25 * np.random.dirichlet(0.3 * np.ones(len(next_move_probabilities)))
            next_move = Move(free_points[np.random.choice(len(free_points), p=next_move_probabilities)])
            self._experience_collector.record_decision(root_board_matrix, np.asarray(visit_counts))
        else:
            next_move_probabilities = softmax(1.0/self._temperature*np.log(np.asarray(visit_counts_of_free_points)+1e-10))
            next_move = Move(free_points[np.random.choice(len(free_points), p=next_move_probabilities)])

        
        self._mcts_tree.go_down(game,next_move)
        return next_move
