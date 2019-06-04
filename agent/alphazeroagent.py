import copy
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common.board import Board
from common.encoder import Encoder
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from common.point import Point


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


class MultiplePlaneEncoder(Encoder):
    def __init__(self,num_plane,board_size):
        self._board_size= board_size
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

    def encode(self, game_states):
        board_matrix = np.zeros(self.shape(),dtype=int)
        for plane in range(len(game_states)):
            player_in_action = game_states[plane].player_in_action
            for row in range(self._board_height):
                for col in range(self._board_width):
                    point = Point(row+1, col+1)
                    piece_owner = game_states[plane].board.get_piece_at_point(point).owner
                    if piece_owner is not None:
                        if piece_owner == player_in_action:
                            board_matrix[plane, row, col] = 1
                        else:
                            board_matrix[plane, row, col] = -1
        return board_matrix

    def shape(self):
        return   self._num_plane, self._board_height, self._board_width

    def encode_point(self, point):
        return self._board_width*(point.row-1)+(point.col-1)

    def decode_point_index(self, index):
        row = index // self._board_width
        col = index % self._board_width
        return Point(row=row+1, col=col+1)

    def num_points(self):
        return self._board_width*self._board_height


class AlphaZeroExpericenceBuffer:
    def __init__(self, states,rewards,visit_counts):
        self._states = states
        self._rewards = rewards
        self._visit_counts = visit_counts

    @property
    def states(self):
        return self._states
     

    @property
    def visit_counts(self):
        return self._visit_counts 

    @property
    def rewards(self):
        return self._rewards


    def serialize(self, path):
        torch.save({'states': self._states,'rewards': self._rewards,'visit_counts': self.visit_counts}, path)

    def deserialize(self, path):
        saved = torch.load(path)
        return AlphaZeroExpericenceBuffer(saved['states'], saved['rewards'], saved['visit_counts'])

    @staticmethod
    def combine_experience(collectors):
        combined_states = np.concatenate([np.array(c.states) for c in collectors])
        combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
        combined_visit_counts = np.concatenate([np.array(c.visit_count) for c in collectors])
        return AlphaZeroExpericenceBuffer(combined_states,combined_rewards,combined_visit_counts)
    




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
    def visit_count(self):
        return self._visit_counts

    @property
    def rewards(self):
        return self._rewards

   

class Game_State_Memory:
    def __init__(self, capacity):
        self._capacity = capacity
        self._game_states = deque()

    def push(self, experience):
        self._game_states.append(experience)
        if self.size() > self._capacity:
            self._game_states.popleft()
  
    def size(self):
        return len(self._game_states)

    @property 
    def game_states(self):
        return list(self._game_states)

class AlphaZeroAgent(Player):
    def __init__(self, encoder, model, num_rounds, experience_collector=None):
        self._encoder = encoder
        self._model = model
        self._num_rounds = num_rounds
        self._experience_collector = experience_collector
        self._game_state_memory= Game_State_Memory(10)

    @property
    def experience_collector(self):
        return self._experience_collector

    def set_experience_collector(self,experience_collector):
        self._experience_collector = experience_collector

    def create_node(self, game_state, parent_branch=None, parent_node=None):
        board_matrix = self._encoder.encode(game_state)
        estimated_branch_priors, estimated_state_value = self._model(
            board_matrix)
        chiildren_branch = {}
        for idx, p in enumerate(estimated_branch_priors):
            point = self._encoder.decode_point_index(idx)
            if game_state.board.is_free_point(point):
                chiildren_branch[point] = Branch(Move(point), p)

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

        self._game_state_memory.push(game_state)
        board_matrix = self._encoder.encode(self._game_state_memory.game_states)
        visit_counts = np.array([root.visit_count(self._encoder.decode_point_index(idx)) for idx in range(self._encoder.num_points())])
        self._experience_collector.record_decision(board_matrix, visit_counts)

        return max(root.children_branch(), key=root.visit_count)


    @classmethod
    def train(cls, expeience, model,learning_rate, batch_size):
        
        model.train()

        criterion_policy = nn.CrossEntropyLoss()
        criterion_value = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(),lr=learning_rate) 

        
        num_examples = expeience.states.shape[0]
        model_input = expeience.states

        visit_sums = np.sum(expeience.visit_counts,axis=1).reshape((num_examples, 1))
        action_policy_target = expeience.visit_counts / visit_sums

        value_target = expeience.rewards
        [action_policy, value] = model(model_input)

        loss_policy = criterion_policy(action_policy, action_policy_target)
        loss_value =  criterion_value(value, value_target)

        optimizer.zero_grad()
        loss= loss_policy + loss_value
        
        loss.backward()
        optimizer.step()
    
    
        