import copy
import gc
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from common.board import Board
from common.encoder import Encoder
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from common.point import Point


class Branch:
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
    def parent_node(self):
        return self._parent_node

    @property
    def child_node(self):
        return self._child_node    
   
    @child_node.setter
    def child_node(self, node):
        self._child_node = node
        
    

class Node:
    def __init__(self, game_state, game_state_value,parent_branch,temperature=0.8):
        self._game_state = game_state
        self._game_state_value = game_state_value
        self._total_visit_counts = 1

        self._parent_branch = parent_branch
        self._children_branch = None
       
        self._temperature = temperature

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
        return self._temperature
    
    
    @property
    def children_branch(self):
        return self._children_branch.keys()

    @children_branch.setter
    def children_branch(self, value):
        self._children_branch = value   

    def expected_value_of_branch(self, point):
        branch = self._children_branch[point]
        if branch.visit_counts == 0:
            return 0.0
        return branch.total_value/branch.visit_counts

    def prior_of_branch(self, point):
        return self._children_branch[point].prior

    def visit_counts_of_branch(self, point):
        if point in self._children_branch:
            return self._children_branch[point].visit_counts
        return 0
       

    def record_visit(self, point, value):
        self._total_visit_counts += 1
        self._children_branch[point].visit_counts += 1
        self._children_branch[point].total_value += value

    def select_branch(self, is_root=False,is_selfplay=True):
        if not self.children_branch:    # no free points to move 
            return None
                
        Qs = [self.expected_value_of_branch(point) for point in self.children_branch]
        Ps = [self.prior_of_branch(point) for point in self.children_branch]
        Ns = [self.visit_counts_of_branch(point) for point in self.children_branch]

        if is_root and is_selfplay:
            noises = np.random.dirichlet([0.03] * len(self.children_branch))
            Ps = [0.75*p+0.25*noise for p, noise in zip(Ps, noises)]

        scores = [(q + self._temperature * p * np.sqrt(self._total_visit_counts) / (n + 1)).item() for q, p, n in zip(Qs, Ps, Ns)]
        best_point_index = np.argmax(scores)

        points = list(self.children_branch)
        return self._children_branch[points[best_point_index]]
        

class MultiplePlaneEncoder(Encoder):
    def __init__(self, num_plane, board_size):
        self._board_size = board_size
        self._board_width = board_size
        self._board_height = board_size
        self._num_plane = num_plane

    def name(self):
        return 'MultiplePlaneEncoder'

    @property
    def num_plane(self):
        return self._num_plane

    @property
    def board_width(self):
        return self._board_width

    @property
    def board_height(self):
        return self._board_height

    def encode(self, boards):
        board_matrix = np.zeros(self.shape(), dtype=int)
        for plane in range(len(boards)):
            for row in range(self._board_height):
                for col in range(self._board_width):
                    point = Point(row+1, col+1)
                    piece = boards[plane].get_piece_at_point(point)
                    if piece is not None:
                        board_matrix[plane, row, col] = piece.owner_id
        return board_matrix

    def shape(self):
        return self._num_plane, self._board_height, self._board_width

    def encode_point(self, point):
        return self._board_width*(point.row-1)+(point.col-1)

    def decode_point_index(self, index):
        row = index // self._board_width
        col = index % self._board_width
        return Point(row=row+1, col=col+1)

    def num_points(self):
        return self._board_width*self._board_height


class AlphaZeroExpericenceBuffer:
    def __init__(self,compacity):
        self._data = deque(maxlen=compacity)

    @property 
    def data(self):
        return self._data

    def combine_experience(self,collectors):
        combined_states = np.concatenate([np.array(c.states) for c in collectors])
        combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
        combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])

        zipped_data= zip(combined_states,combined_rewards,combined_visit_counts)
        self._data.extend(zipped_data)

    def size(self):
        return len(self._data)
   

class AlphaZeroExperienceCollector:
    def __init__(self):
        self._states = []
        self._visit_counts = []
        self._rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def reset_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self._states += self._current_episode_states
        self._visit_counts += self._current_episode_visit_counts
        self._rewards += [reward for _ in range(num_states)]
        self.reset_episode()

    @property
    def visit_counts(self):
        return self._visit_counts

    @property
    def rewards(self):
        return self._rewards

    @property
    def states(self):
        return self._states


class AlphaZeroAgent(Player):
    def __init__(self, id, name, mark, encoder, model, num_rounds,temperature,experience_collector=None, device='cpu'):
        super().__init__(id, name, mark)
        self._encoder = encoder
        self._device =  device
        self._model =   model
        self._num_rounds = num_rounds
        self._experience_collector = experience_collector
        self._temperature = temperature

    def reset(self):
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

    def create_node(self, game_state, estimated_branch_priors,estimated_state_value,parent_branch=None):
       
        new_node = Node(game_state, estimated_state_value,parent_branch,self._temperature)
               
        chiildren_branch = {}
        for idx, p in enumerate(estimated_branch_priors):
            point = self._encoder.decode_point_index(idx)
            if game_state.board.is_free_point(point):
                chiildren_branch[point] = Branch(new_node,Move(point), p)
        
        new_node.children_branch= chiildren_branch
                
        return new_node
      

    def select_move(self, game, game_state):  # it is guaranteed that it is not the final game state anyway
        game.state_cache.push(game_state.board)
        root_board_matrix = self._encoder.encode( game.state_cache.game_states)
        model_input = torch.from_numpy(root_board_matrix).unsqueeze(0).to(self._device, dtype=torch.float)
        estimated_branch_priors, estimated_state_value = self.predict(model_input)
        root = self.create_node(game_state, estimated_branch_priors[0], estimated_state_value[0].item())

        for _ in tqdm(range(self._num_rounds)):
            node = root
            game_state_memory = copy.deepcopy(game.state_cache)

            # select
            next_branch = node.select_branch(is_root=True,is_selfplay=game.is_selfplay)
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
                    next_branch = None


            #backup
            value = 0
            if game.is_final_state(node.game_state):
                value = -1
            else:
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

        
    

