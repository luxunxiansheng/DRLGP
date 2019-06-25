import copy
import gc
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

    @total_value.setter
    def total_value(self, value):
        self._total_value = value

    @property
    def visit_count(self):
        return self._visit_count

    @visit_count.setter
    def visit_count(self, value):
        self._visit_count = value


class Node:

    temperature = 0.8

    def __init__(self, game_state, game_state_value, children_branch,parent_branch,parent_node):
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
    def parent_branch(self):
        return self._parent_branch

    @property
    def parent_node(self):
        return self._parent_node

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

    
    
    def score_root_node_branch(self, point):
        q = self.expected_value_of_branch(point)
        p = self.prior_of_branch(point)
        n = self.visit_count_of_branch(point)

        score = (q + Node.temperature * p * np.sqrt(self._total_visit_count) / (n + 1)).item()
        #print("{:d}-{:d}: q {:.4f} p {:.4f} n {:d} score {:.4f}".format(point.row,point.col,q,p,n,score))

        return score
    
   

    def select_branch(self, is_root=False):
        points = [point for point in self.children_branch()] 
        Qs = [self.expected_value_of_branch(point) for point in self.children_branch()]
        Ps = [self.prior_of_branch(point) for point in self.children_branch()]
        Ns = [self.visit_count_of_branch(point) for point in self.children_branch() ] 
        
        if is_root:
            noises = np.random.dirichlet([0.03] * len(self.children_branch()))
            Ps= [0.75*p+0.25*noise for p, noise in zip(Ps,noises)]
           
        scores=[(q + Node.temperature * p * np.sqrt(self._total_visit_count) / (n + 1)).item() for  q,p,n in zip(Qs,Ps,Ns)]
        best_point_index = np.argmax(scores)
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
    def __init__(self, states, rewards, visit_counts):
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
        torch.save({'states': self._states, 'rewards': self._rewards, 'visit_counts': self._visit_counts}, path)

    @classmethod
    def deserialize(cls, path):
        saved = torch.load(path)
        return AlphaZeroExpericenceBuffer(saved['states'], saved['rewards'], saved['visit_counts'])

    @staticmethod
    def combine_experience(collectors):
        combined_states = np.concatenate([np.array(c.states) for c in collectors])
        combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
        combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
        return AlphaZeroExpericenceBuffer(combined_states, combined_rewards, combined_visit_counts)

    def size(self):
        return self._states.shape[0]


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

    def clear(self):
        self._game_states.clear()


class AlphaZeroAgent(Player):

    def __init__(self, id, name, mark, encoder, model, num_rounds, experience_collector=None, device='cpu'):
        super().__init__(id, name, mark)
        self._encoder = encoder
        self._device = device
        self._model = model.to(device)
        self._num_rounds = num_rounds
        self._experience_collector = experience_collector
        self._game_state_memory = Game_State_Memory(10)

    def reset_memory(self):
        self._game_state_memory.clear()

    @property
    def experience_collector(self):
        return self._experience_collector

    def set_experience_collector(self, experience_collector):
        self._experience_collector = experience_collector

    def create_node(self, game_state, estimated_branch_priors, estimated_state_value, parent_branch=None, parent_node=None):
        chiildren_branch = {}
        for idx, p in enumerate(estimated_branch_priors):
            point = self._encoder.decode_point_index(idx)
            if game_state.board.is_free_point(point):
                chiildren_branch[point] = Branch(Move(point), p)

        new_node = Node(game_state, estimated_state_value, chiildren_branch,parent_branch,parent_node)
        if parent_node is not None:
            parent_node.add_child_node(parent_branch.move.point, new_node)
        return new_node

    def select_move(self, game, game_state):
        self._game_state_memory.push(game_state.board)
        root_board_matrix = self._encoder.encode(self._game_state_memory.game_states)
        model_input = torch.from_numpy(root_board_matrix).unsqueeze(0).to(self._device, dtype=torch.float)
        estimated_branch_priors, estimated_state_value = self.predict(model_input)
        root = self.create_node(game_state, estimated_branch_priors[0], estimated_state_value[0].item())

        for _ in tqdm(range(self._num_rounds)):
            node = root
            game_state_memory = copy.deepcopy(self._game_state_memory)

            # select
            next_branch = node.select_branch(True)
            while node.has_child_node(next_branch.move.point):
                node = node.get_child_node(next_branch.move.point)
                game_state_memory.push(node.game_state.board)
                next_branch = node.select_branch()

            # expand
            new_state = game.transit(node.game_state, next_branch.move)
            game_state_memory.push(new_state.board)

            parent_branch = next_branch
            parent_node = node

            temp_board_matrix = self._encoder.encode(game_state_memory.game_states)
            model_input = torch.from_numpy(temp_board_matrix).unsqueeze(0).to(self._device, dtype=torch.float)
            estimated_branch_priors, estimated_state_value = self.predict(model_input)

            new_node = self.create_node(new_state, estimated_branch_priors[0], estimated_state_value[0].item(),parent_branch,parent_node)
            
            # backup
            value = -1*new_node.game_state_value
            while parent_node is not None:
                parent_node.record_visit(parent_branch.move.point, value)
                parent_branch = parent_node.parent_branch
                parent_node = parent_node.parent_node
                value = -1 * value

        visit_counts = np.array([root.visit_count_of_branch(self._encoder.decode_point_index(idx)) for idx in range(self._encoder.num_points())])
        self._experience_collector.record_decision(root_board_matrix, visit_counts)

        return Move(max(root.children_branch(), key=root.visit_count_of_branch))

    def predict(self, input_states):
        return self._model(input_states)

    @classmethod
    def train(cls, expeience, model, learning_rate, batch_size, device, writer):
        model = model.to(device)
        

        criterion_policy = nn.KLDivLoss()
        criterion_value = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        num_examples = expeience.size()
        
        

        for i in tqdm(range(int(num_examples / batch_size))):
            states = torch.from_numpy(expeience.states[i*batch_size:(i+1)*batch_size]).to(device, dtype=torch.float)
            visit_counts = torch.from_numpy(expeience.visit_counts[i*batch_size:(i+1)*batch_size]).to(device)
            rewards = torch.from_numpy(expeience.rewards[i*batch_size:(i+1)*batch_size]).to(device, dtype=torch.float).unsqueeze(1)

            visit_sums = visit_counts.sum(dim=1).view((states.shape[0], 1))
            action_policy_target = visit_counts.float() / visit_sums.float()

            value_target = rewards

            [action_policy, value] = model(states)

            loss_policy = criterion_policy(action_policy, action_policy_target)
            loss_value = criterion_value(value, value_target)

            optimizer.zero_grad()
            loss = loss_policy + loss_value

            print(loss.item())

            writer.add_scalar('loss', loss.item(), i)

            loss.backward()
            optimizer.step()
