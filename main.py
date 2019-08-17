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


import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.ResNet8Network import ResNet8Network
from models.Simple5Network import Simple5Network
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent.alphazeroagent.alphazeroagent import AlphaZeroAgent
from agent.alphazeroagent.experiencebuffer import ExpericenceBuffer
from agent.alphazeroagent.experiencecollector import ExperienceCollector
from agent.alphazeroagent.mcts.tree import Tree
from agent.mctsagent import MCTSAgent
from boardencoder.snapshotencoder import SnapshotEncoder
from common.board import Board
from common.utils import Utils
from game.connect5game import Connect5Game


class Trainer(object):
    def __init__(self, args, logger):

        self._logger = logger

        config_name = args.config

        # hardcode the config path just for convinence.
        cfg = Utils.config("./config/" + config_name)

        self._number_of_planes = cfg['GAME'].getint('number_of_planes')
        self._board_size = cfg['GAME'].getint('board_size')

        self._az_mcts_round_per_moves = cfg['AZ_MCTS'].getint(
            'round_per_moves')
        self._az_mcts_temperature = cfg['AZ_MCTS'].getfloat('temperature')

        self._basic_mcts_temperature = cfg['BASIC_MCTS'].getfloat(
            'temperature')
        self._basic_mcts_round_per_moves = cfg['BASIC_MCTS'].getint(
            'round_per_moves')

        self._train_number_of_games = cfg['TRAIN'].getint('number_of_games')
        self._batch_of_self_play = cfg['TRAIN'].getint('batch_of_self_play')

        self._buffer_size = cfg['TRAIN'].getint('buffer_size')
        self._learning_rate = cfg['TRAIN'].getfloat('learning_rate')
        self._batch_size = cfg['TRAIN'].getint('batch_size')

        self._epochs = cfg['TRAIN'].getint('epochs')
        self._kl_threshold = cfg['TRAIN'].getfloat('kl_threshold')
        self._check_frequence = cfg['TRAIN'].getint('check_frequence')

        self._current_model_file = './checkpoints/' + \
            config_name.split('.')[0]+'/current.model'
        self._best_model_file = './checkpoints/' + \
            config_name.split('.')[0]+'/best.model'

        self._evaluate_number_of_games = cfg['EVALUATE'].getint(
            'number_of_games')
        self._multipleprocessing = cfg['EVALUATE'].getboolean(
            'mutipleprocessing')

        os.makedirs(os.path.dirname(self._current_model_file), exist_ok=True)
        os.makedirs(os.path.dirname(self._best_model_file), exist_ok=True)

        use_cuda = torch.cuda.is_available()

        if use_cuda:
            gpu_ids = list(map(int, args.gpu_ids.split(',')))
            num_devices = torch.cuda.device_count()
            if num_devices < len(gpu_ids):
                raise Exception(
                    '#available gpu : {} < --device_ids : {}'.format(num_devices, len(gpu_ids)))
            cuda = 'cuda:' + str(gpu_ids[0])
        self._device = torch.device(cuda if use_cuda else 'cpu')

        self._encoder = SnapshotEncoder(self._number_of_planes, self._board_size)

        input_shape = (self._number_of_planes, self._board_size, self._board_size)
        model_name = cfg['MODELS'].get('net')
        self._model = ResNet8Network(input_shape, self._board_size * self._board_size) if model_name == 'ResNet8Network' else Simple5Network(
            input_shape, self._board_size * self._board_size)

        # Be aware this is not the first time to run this program
        resume = args.resume
        if resume:
            self._model.load_state_dict(torch.load(self._current_model_file))

        #self._model = DataParallel(self._model,device_ids=gpu_ids)
        self._model = self._model.to(self._device)

        self._experience_buffer = ExpericenceBuffer(self._buffer_size)

        self._optimizer = Utils.get_optimizer(self._model.parameters(), cfg)

        self._writer = SummaryWriter(
            log_dir='./runs/'+config_name.split('.')[0])

    @staticmethod
    def _collect_data_once_in_parallel(encoder, model, az_mcts_round_per_moves, board_size, number_of_planes, device, pipe):
        mcts_tree = Tree()
        agent_1 = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "AlphaZeroAgent1", encoder, model, mcts_tree, az_mcts_round_per_moves, device=device)
        agent_2 = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "AlphaZeroAgent2", encoder, model, mcts_tree, az_mcts_round_per_moves, device=device)

        board = Board(board_size)
        players = [agent_1, agent_2]
        game = Connect5Game(board, players,  players[random.choice([0, 1])], number_of_planes, True)

        while not game.is_over():
            move = game.working_game_state.player_in_action.select_move(game)
            game.apply_move(move)

            # game.working_game_state.board.print_board()

        # game.working_game_state.board.print_board()

        winner = game.final_winner

        if winner is not None:
            if winner == players[0]:
                players[0].experience_collector.complete_episode(reward=1)
                players[1].experience_collector.complete_episode(reward=-1)
            if winner == players[1]:
                players[1].experience_collector.complete_episode(reward=1)
                players[0].experience_collector.complete_episode(reward=-1)

            expericence_buffer = ExpericenceBuffer()
            expericence_buffer.combine_experience([agent_1.experience_collector, agent_2.experience_collector])
            pipe.send(expericence_buffer)
            
            pipe.close()

    def _collect_data_in_parallel(self):

        processes = []
        pipes = []

        for _ in range(self._batch_of_self_play):
            parent_connection_end, child_connection_end = mp.Pipe()

            p = mp.Process(target=Trainer._collect_data_once_in_parallel, args=(self._encoder, self._model,
                                                                                self._az_mcts_round_per_moves, self._board_size, self._number_of_planes, self._device, child_connection_end))
            processes.append(p)
            pipes.append((parent_connection_end,child_connection_end))
            p.start()

        for (parent_connection_end,child_connection_end) in pipes:
            child_connection_end.close()
            experience_buffer = parent_connection_end.recv()
            self._experience_buffer.merge(experience_buffer)

        for p in processes:
            p.join()
        for (parent_connection_end,_) in pipes:
            parent_connection_end.close()

        print('------------------------------------------------------------------------------------')
        print(self._experience_buffer.size())
        

    def _collect_data_once(self):

        mcts_tree = Tree()
        agent_1 = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "AlphaZeroAgent1", self._encoder, self._model, mcts_tree, self._az_mcts_round_per_moves, device=self._device)
        agent_2 = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "AlphaZeroAgent2", self._encoder, self._model, mcts_tree, self._az_mcts_round_per_moves, device=self._device)

        board = Board(self._board_size)
        players = [agent_1, agent_2]
        game = Connect5Game(board, players,  players[random.choice([0, 1])], self._number_of_planes, True)

        while not game.is_over():
            move = game.working_game_state.player_in_action.select_move(game)
            game.apply_move(move)

            # game.working_game_state.board.print_board()

        # game.working_game_state.board.print_board()
        # self._logger.info(game.final_winner.name)

        winner = game.final_winner
        if winner is not None:
            if winner == players[0]:
                players[0].experience_collector.complete_episode(reward=1)
                players[1].experience_collector.complete_episode(reward=-1)
            if winner == players[1]:
                players[1].experience_collector.complete_episode(reward=1)
                players[0].experience_collector.complete_episode(reward=-1)

            self._experience_buffer.combine_experience([agent_1.experience_collector, agent_2.experience_collector])

    def _collect_data(self, batch_index):
        if self._multipleprocessing:
            self._collect_data_in_parallel()
        else:
            for _ in range(self._batch_of_self_play):
                self._collect_data_once()

    def _improve_policy(self, game_index):
        self._model.train()

        batch_data = random.sample(
            self._experience_buffer.data, self._batch_size)

        for _ in range(self._epochs):
            states, rewards, visit_counts = zip(*batch_data)

            states = torch.from_numpy(np.array(list(states))).to(
                self._device, dtype=torch.float)
            rewards = torch.from_numpy(np.array(list(rewards))).to(
                self._device, dtype=torch.float)
            visit_counts = torch.from_numpy(np.array(list(visit_counts))).to(
                self._device, dtype=torch.float)

            action_policy_target = F.softmax(visit_counts, dim=1)
            value_target = rewards

            [action_policy, value] = self._model(states)

            log_action_policy = torch.log(action_policy)
            loss_policy = - log_action_policy * action_policy_target
            loss_policy = loss_policy.sum(dim=1).mean()

            loss_value = F.mse_loss(value.squeeze(), value_target)

            loss = loss_policy + loss_value

            with torch.no_grad():
                entroy = - \
                    torch.mean(
                        torch.sum(log_action_policy*action_policy, dim=1))

            self._writer.add_scalar('loss', loss.item(), game_index)
            self._writer.add_scalar('loss_value', loss_value.item(), game_index)
            self._writer.add_scalar('loss_policy', loss_policy.item(), game_index)
            self._writer.add_scalar('entropy', entroy.item(), game_index)

            self._optimizer.zero_grad()

            loss.backward()
            self._optimizer.step()
            [updated_action_policy, _] = self._model(states)
            kl = F.kl_div(action_policy, updated_action_policy).item()

            if kl > self._kl_threshold * 4:
                break

    def _evaluate_policy_once(self):

        mcts_tree = Tree()
        mcts_agent = MCTSAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "MCTSAgent", self._basic_mcts_round_per_moves, self._basic_mcts_temperature)
        az_agent = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "AlphaZeroAgent", self._encoder, self._model, mcts_tree, self._az_mcts_round_per_moves, device=self._device)

        board = Board(self._board_size)
        players = [mcts_agent, az_agent]

        game = Connect5Game(board, players, players[random.choice([0, 1])], self._number_of_planes, is_self_play=False)

        while not game.is_over():
            move = game.working_game_state.player_in_action.select_move(game)
            game.apply_move(move)

            # game.working_game_state.board.print_board()

        # game.working_game_state.board.print_board()
        self._logger.info(game.final_winner.name)

        winner = game.final_winner

        if winner is not None:
            return 1 if winner.id == az_agent.id else - 1

        return 0

    @staticmethod
    def _evaluate_policy_once_in_parallel(basic_mcts_round_per_moves, basic_mcts_temperature, encoder, model, az_mcts_round_per_moves, device, board_size, number_of_planes, pipe):
        mcts_tree = Tree()
        mcts_agent = MCTSAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "MCTSAgent", basic_mcts_round_per_moves, basic_mcts_temperature)
        az_agent = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "AlphaZeroAgent", encoder, model, mcts_tree, az_mcts_round_per_moves, device=device)

        board = Board(board_size)
        players = [mcts_agent, az_agent]

        game = Connect5Game(board, players, players[random.choice([0, 1])], number_of_planes, is_self_play=False)

        while not game.is_over():
            move = game.working_game_state.player_in_action.select_move(game)
            game.apply_move(move)

            # game.working_game_state.board.print_board()

        # game.working_game_state.board.print_board()

        winner = game.final_winner

        score = 0
        if winner is not None:
            score = 1 if winner.id == az_agent.id else -1

        pipe.send(score)
        pipe.close()

    def _evaluate_ploicy_in_parallel(self):

        final_score = 0

        processes = []
        pipes = []

        for _ in range(self._evaluate_number_of_games):
            parent_connection_end, child_connection_end = mp.Pipe()

            p = mp.Process(target=Trainer._evaluate_policy_once_in_parallel, args=(self._basic_mcts_round_per_moves, self._basic_mcts_temperature, self._encoder,
                                                                                   self._model, self._az_mcts_round_per_moves, self._device, self._board_size, self._number_of_planes, child_connection_end))

            processes.append(p)
            pipes.append(parent_connection_end)
            p.start()

        for parent_connection_end in pipes:
            final_score += parent_connection_end.recv()

        for p in processes:
            p.join()
        for pipe in pipes:
            pipe.close()

        return final_score

    def _evaluate_policy(self):

        final_score = 0

        if self._multipleprocessing:
            final_score = self._evaluate_ploicy_in_parallel()
        else:
            for _ in tqdm(range(self._evaluate_number_of_games)):
                final_score += self._evaluate_policy_once()

        self._logger.info('Alphazero gets {} in {}'.format(final_score, self._evaluate_number_of_games))

        return final_score

    def run(self):

        mp.set_start_method('spawn')

        best_score = 0

        for batch_index in tqdm(range(1, int(self._train_number_of_games/self._batch_of_self_play))):
            # collect data via self-playing
            self._collect_data(batch_index)

            if self._experience_buffer.size() > self._batch_size:
                # update the policy with SGD
                self._improve_policy(batch_index)

            if batch_index % self._check_frequence == 0:

                score = self._evaluate_policy()

                self._logger.info("current self-play batch:{} and score  is:{}".format(batch_index, score))

                torch.save(self._model.state_dict(), self._current_model_file)

                if score > best_score:
                    self._logger.info("New best policy!!!!!!!!")
                    best_score = score
                    # update the best_policy
                    torch.save(self._model.state_dict(), self._best_model_file)
                    if (best_score == self._evaluate_number_of_games and self._basic_mcts_round_per_moves < 6000):
                        self._basic_mcts_round_per_moves += 1000
                        best_score = 0


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help="Specifiy which gpu devices to use if available,e.g. '0,1,2'")
    parser.add_argument('--resume', type=bool, default=False,
                        help='Wethere resume traning from the previous or not ')
    parser.add_argument('-config', type=str, default='default.ini',
                        help='A ini config file to setup the default machinery')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    Trainer(args, logger).run()


if __name__ == '__main__':
    main()
