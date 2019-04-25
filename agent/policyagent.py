import random

import numpy as np
import pysnooper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from common.board import Board
from common.encoder import Encoder
from common.experiencecollector import ExperienceCollector
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from models.feedfowrdnerualnetwork import FeedForwardNeuralNetwork


class PolicyAgent(Player):
    def __init__(self, id, name, mark, encoder, collector, state_dict=None):
        super().__init__(id, name, mark)
        self._model = FeedForwardNeuralNetwork()
        if state_dict is not None:
            self._model.load_state_dict(state_dict)
        self._encoder = encoder
        self._board_size = encoder.board_width*encoder.board_height
        self._num_points = self._board_size
        self._epsilon = 0.1
        self._collector = collector

  
    # @pysnooper.snoop()
    def select_move(self, game, game_state):
        # epsilon greedy
        if np.random.random() < self._epsilon:
            point_probs = (np.ones(self._num_points)/self._num_points).reshape(1,self._num_points)
        else:
            point_probs = self._predict(game_state).detach().numpy()

        point_probs = self._clip_probs(point_probs)

        possible_points = np.arange(self._num_points)
        ranked_points = np.random.choice(possible_points, size=self._num_points, replace=False, p=point_probs[0])
        for point_index in ranked_points:
            point = self._encoder.decode_point_index(point_index)
            if game_state.board.is_valid_point(point):
                if self._collector is not None:
                    self._collector.record_decision(
                        state=game_state, action=point)

                return Move(point)

    def _predict(self, game_state):
        board_matrix = self._encoder.encode(game_state)
        board_matrix = board_matrix.reshape([1, self._num_points])

        return self._model(torch.tensor(board_matrix, dtype=torch.float))

    def _clip_probs(self, original_probs):
        min_p = 1e-5
        max_p = 1 - min_p

        clipped_probs = np.clip(original_probs, min_p, max_p)
        clipped_probs = clipped_probs/np.sum(clipped_probs)

        return clipped_probs

    def train(self, experience):
        torch.manual_seed(1)
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        batch_size = 32

        optimizer = optim.SGD(self._model.parameters(),
                              lr=0.1, momentum=0.5, weight_decay=0.01)

        num_sample = experience.states.shape[0]
        num_moves = self._board_size

        X = experience.states
        Y = np.zeros((num_sample, num_moves))
        for i in range(num_sample):
            action = experience.actions[i]
            reward = experience.rewards[i]
            Y[i][action] = reward

        train_data = [[board, move] for board, move in zip(X, Y)]

        random.shuffle(train_data)
        train_batches = [train_data[k:k+batch_size]
                         for k in range(0, num_sample, batch_size)]

        self._model.train()

        for _, mini_batch in tqdm(enumerate(train_batches)):
            train_correct += self._train_batch(optimizer, mini_batch, device)
        print('Train Accuracy:{:.0f}%'.format(100.*train_correct/num_sample))

    def _train_batch(self, optimizer, mini_batch, device='cpu'):
        train_loss = 0
        train_correct = 0

        optimizer.zero_grad()

        X = torch.tensor(np.array(mini_batch)[
                         :, 0, :], dtype=torch.float).to(device)
        Y = torch.tensor(np.array(mini_batch)[
                         :, 1, :], dtype=torch.float).to(device)

        output = self._model(X)
        train_loss = F.mse_loss(output, Y, reduction='sum')

        pred = output.argmax(dim=1, keepdim=True)
        target = Y.argmax(dim=1, keepdim=True)

        train_correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss.backward()
        optimizer.step()

        return train_correct

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        model = FeedForwardNeuralNetwork()
        model.load_state_dict(torch.load(path))
        return model
