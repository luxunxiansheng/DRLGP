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


import random

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from agent.alphazeroagent import AlphaZeroAgent
from agent.experiencebuffer import ExpericenceBuffer
from common.board import Board
from common.gamestate import GameState
from game.connect5game import Connect5Game


class DataCollector:
    def __init__(self, encoder, model, az_mcts_round_per_moves, c_puct, az_mcts_temperature, boder_size, number_of_planes, expericence_buffer, devices_id, use_cuda, logger):
        self._encoder = encoder
        self._model = model
        self._az_mcts_rounds_per_move = az_mcts_round_per_moves
        self._c_puct = c_puct
        self._az_mcts_temperature = az_mcts_temperature
        self._board_size = boder_size
        self._number_of_planes = number_of_planes
        self._experience_buffer = expericence_buffer
        self._logger = logger

        if use_cuda:
            self._devices = [torch.device(
                'cuda:'+str(devices_id[i])) for i in range(len(devices_id))]
        else:
            self._devices = [torch.device('cpu')]

    def collect_data(self, game_index):
        self._model.eval()

        if len(self._devices) > 1:
            self._collect_data_in_parallel()
        else:
            self._collect_data_once()

        self._logger.debug('--Data Collected in round {}--'.format(game_index))

    def _collect_data_once(self):

        agent_1 = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "AlphaZeroAgent1", self._encoder, self._model,
                                 self._az_mcts_rounds_per_move, self._c_puct, self._az_mcts_temperature, device=self._devices[0])
        agent_2 = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "AlphaZeroAgent2", self._encoder, self._model,
                                 self._az_mcts_rounds_per_move, self._c_puct, self._az_mcts_temperature, device=self._devices[0])

        # Two agents use the same tree to save the memory and improve the efficency
        agent_2.mcts_tree = agent_1.mcts_tree

        board = Board(self._board_size)
        players = {agent_1.id: agent_1, agent_2.id: agent_2}
        start_game_state = GameState(
            board, random.choice([agent_1.id, agent_2.id]), None)
        game = Connect5Game(start_game_state, [
                            agent_1.id, agent_2.id], self._number_of_planes, is_self_play=True)
        while not game.is_over():
            move = players[game.working_game_state.player_in_action].select_move(
                game)
            game.apply_move(move)

            # game.working_game_state.board.print_board()

        # game.working_game_state.board.print_board()

        winner = game.final_winner
        if winner is not None:
            if winner == players[0].id:
                players[0].experience_collector.complete_episode(reward=1)
                players[1].experience_collector.complete_episode(reward=-1)
            if winner == players[1].id:
                players[1].experience_collector.complete_episode(reward=1)
                players[0].experience_collector.complete_episode(reward=-1)

            self._experience_buffer.combine_experience(
                [agent_1.experience_collector, agent_2.experience_collector])

        self._logger.debug("buffer size is :{}".format(
            self._experience_buffer.size()))

    @staticmethod
    def _collect_data_once_in_parallel(encoder, model, az_mcts_round_per_moves, board_size, number_of_planes, c_puct, az_mcts_temperature, device, pipe):

        agent_1 = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "AlphaZeroAgent1", encoder,
                                 model, az_mcts_round_per_moves, c_puct, az_mcts_temperature, device=device)
        agent_2 = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "AlphaZeroAgent2", encoder,
                                 model, az_mcts_round_per_moves, c_puct, az_mcts_temperature, device=device)
        agent_2.mcts_tree = agent_1.mcts_tree

        board = Board(board_size)
        players = {agent_1.id: agent_1, agent_2.id: agent_2}
        start_game_state = GameState(
            board, random.choice([agent_1.id, agent_2.id]), None)
        game = Connect5Game(start_game_state, [agent_1.id, agent_2.id], number_of_planes, is_self_play=True)

        while not game.is_over():
            move = players[game.working_game_state.player_in_action].select_move(
                game)
            game.apply_move(move)

            # game.working_game_state.board.print_board()

        # game.working_game_state.board.print_board()

        winner = game.final_winner

        if winner is not None:
            if winner == players[0].id:
                players[0].experience_collector.complete_episode(reward=1)
                players[1].experience_collector.complete_episode(reward=-1)
            if winner == players[1].id:
                players[1].experience_collector.complete_episode(reward=1)
                players[0].experience_collector.complete_episode(reward=-1)

            expericence_buffer = ExpericenceBuffer()
            expericence_buffer.combine_experience(
                [agent_1.experience_collector, agent_2.experience_collector])
            pipe.send(expericence_buffer)
            pipe.close()

    def _collect_data_in_parallel(self):
        processes = []
        pipes = []

        for gpu_index in range(len(self._devices)):
            parent_connection_end, child_connection_end = mp.Pipe()
            p = mp.Process(target=DataCollector._collect_data_once_in_parallel, args=(self._encoder, self._model, self._az_mcts_rounds_per_move, self._board_size,
                                                                                      self._number_of_planes, self._c_puct, self._az_mcts_temperature,
                                                                                      self._devices[gpu_index], child_connection_end))
            processes.append(p)
            pipes.append((parent_connection_end, child_connection_end))
            p.start()

        for (parent_connection_end, child_connection_end) in pipes:
            child_connection_end.close()

            while True:
                try:
                    experience_buffer = parent_connection_end.recv()
                    self._experience_buffer.merge(experience_buffer)
                except EOFError:

                    break

        for (parent_connection_end, _) in pipes:
            parent_connection_end.close()

        for p in processes:
            p.join()

       