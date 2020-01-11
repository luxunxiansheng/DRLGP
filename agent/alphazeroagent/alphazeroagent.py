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

class MCTSNode(object):
    def __init__(self, game_state, prior_p, parent=None):
        self._game_state = game_state
        self._parent = parent
        self._children = {}
        self._num_visits = 0
        
        self._Q = 0
        self._P = prior_p
        self._u = 0

    def select(self,c_puct):
        child=max(self._children.items(),key=lambda point_node: point_node[1].get_value(c_puct))
        return child[1]
    
    def update_recursively(self,root_node,leaf_value):
        if self!=root_node and self._parent:
            self._parent.update_recursively(root_node,-leaf_value)
                
        self._num_visits +=1
        self._Q += 1.0*(leaf_value-self._Q)/self._num_visits

    def is_leaf(self):
        return self._children == {}
    
    @property
    def num_visits(self):
        return self._num_visits

    @property
    def game_state(self):
        return self._game_state

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent
    
    @parent.setter
    def parent(self,value):
        self._parent = value

    def get_child(self,point):
        return self._children.get(point)

    def add_child(self,game,new_point,prior):
        new_game_state = game.look_ahead_next_move(self._game_state, Move(new_point))
        new_node = MCTSNode(new_game_state,prior,self)
        self._children[new_point]=new_node
        return new_node

    def get_value(self,c_puct):
        """
        Same as in alphazero
        """
        self._u = (c_puct * self._P *np.sqrt(self._parent._num_visits) / (1 + self._num_visits))
        return self._Q + self._u


class MCTSTree(object):
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
    
    def go_down(self,game,move):
        if self._working_node is not None:
            if move.point in self._working_node.children:
                child=self.working_node.children.pop(move.point)
                child.parent= None  
            else:
                if not self._working_node.is_terminal(game):
                    new_game_state = game.look_ahead_next_move(self._working_node.game_state, Move(new_point))
                    child = MCTSNode(new_game_state,1.0,None)
                else:
                    child = None
            self._working_node = child

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
            self._mcts_tree.working_node = MCTSNode(game.working_game_sate,1.0,None)

        for _ in tqdm(range(self._num_rounds), desc='Rollout Loop'):
            node = self._mcts_tree.working_node
            game_state_memory = copy.deepcopy(game.state_cache)

            while True:
                if node.is_leaf():
                    break
                node = node.select(self._temperature)
                game_state_memory.push(node.game_state)


            leaf_value = 0.0 
            if not game.is_final_state(node.game_state):
                # encode  the last specified boards as the root
                board_matrix = self._encoder.encode(game.state_cache.game_states, game.working_game_state.player_in_action, game.working_game_state.previous_move)
                estimated_priors, leaf_value = self._predict(board_matrix) 
                free_points=node.game_state.board.get_legal_points()
                for point in free_points:
                    node.add_child(game,point,estimated_priors[point])
            else:
                if game.final_winner is not None:
                    leaf_value = 1.0 if game.final_winner == node.game_state.player_in_action else -1.0

            node.update_recursively(self._mcts_tree.working_node,-leaf_value)

        self._mcts_tree.working_node.game_state.board.print_visits(self._mcts_tree.working_node.children)   
                    
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
