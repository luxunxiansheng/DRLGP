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

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from agent.alphazeroagent import AlphaZeroAgent
from agent.mctsagent import MCTSAgent
from common.board import Board
from common.gamestate import GameState
from game.connect5game import Connect5Game


class PolicyEvaluator:
    def __init__(self, devices_ids, use_cuda, encoder, board_size, number_of_planes, model, az_mcts_round_per_moves, c_puct, az_mcts_temperature, basic_mcts_c_puct, basic_mcts_round_per_moves, evaluate_number_of_games, logger):

        self._devices_ids = devices_ids

        if use_cuda:
            self._devices = [torch.device(
                'cuda:'+str(devices_ids[i])) for i in range(len(devices_ids))]
        else:
            self._devices = [torch.device('cpu')]

        self._board_size = board_size

        self._model = model
        self._encoder = encoder
        self._number_of_planes = number_of_planes

        self._az_mcts_rounds_per_move = az_mcts_round_per_moves
        self._c_puct = c_puct
        self._az_mcts_temperature = az_mcts_temperature

        self._basic_mcts_c_puct = basic_mcts_c_puct
        self._basic_mcts_rounds_per_move = basic_mcts_round_per_moves

        self._evaluate_number_of_games = evaluate_number_of_games

        self._logger = logger

    def evaluate_policy(self):

        self._model.eval()

        final_score = 0

        if len(self._devices) > 1:
            self._evaluate_number_of_games = len(self._devices)*2
            final_score = self._evaluate_ploicy_in_parallel()
        else:
            for _ in tqdm(range(self._evaluate_number_of_games), desc='Evaluation Loop'):
                final_score += self._evaluate_policy_once()

        self._logger.debug('Alphazero gets win_ratio {:.2%} in {}'.format(
            final_score/self._evaluate_number_of_games, self._evaluate_number_of_games))

        return final_score/self._evaluate_number_of_games

    def _evaluate_policy_once(self):
        device = self._devices[0]
        mcts_agent = MCTSAgent(Connect5Game.ASSIGNED_PLAYER_ID_1, "MCTSAgent",
                               self._basic_mcts_rounds_per_move, self._basic_mcts_c_puct)
        az_agent = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "AlphaZeroAgent", self._encoder, self._model,
                                  self._az_mcts_rounds_per_move, self._c_puct, self._az_mcts_temperature, device=device)

        board = Board(self._board_size)
        players = {}
        players[mcts_agent.id] = mcts_agent
        players[az_agent.id] = az_agent

        start_game_state = GameState(board, mcts_agent.id, None)

        # MCTS agent always plays first
        game = Connect5Game(start_game_state, [mcts_agent.id, az_agent.id], self._number_of_planes, is_self_play=False)

        while not game.is_over():
            move = players[game.working_game_state.player_in_action].select_move(
                game)
            if players[0].id == game.working_game_state.player_in_action:
                players[1].mcts_tree.go_down(game, move)
            else:
                players[0].mcts_tree.go_down(game, move)

            game.apply_move(move)

        winner = game.final_winner

        self._logger.debug('Winner is {}'.format(players[winner].name))

        score = 0
        if winner == az_agent.id:
            score = 1
        return score

    @staticmethod
    def _evaluate_policy_once_in_parallel(basic_mcts_round_per_moves, basic_mcts_c_puct, az_mcts_temperature, encoder, model, az_mcts_round_per_moves, c_puct, device, board_size, number_of_planes, pipe):
        mcts_agent = MCTSAgent(Connect5Game.ASSIGNED_PLAYER_ID_1,
                               "MCTSAgent", basic_mcts_round_per_moves, basic_mcts_c_puct)
        az_agent = AlphaZeroAgent(Connect5Game.ASSIGNED_PLAYER_ID_2, "AlphaZeroAgent", encoder,
                                  model, az_mcts_round_per_moves, c_puct, az_mcts_temperature, device=device)

        board = Board(board_size)
        players = {}
        players[mcts_agent.id] = mcts_agent
        players[az_agent.id] = az_agent

        start_game_state = GameState(board, mcts_agent.id, None)

        # MCTS agent always plays first
        game = Connect5Game(start_game_state, [
                            mcts_agent.id, az_agent.id], number_of_planes, is_self_play=False)

        while not game.is_over():
            move = players[game.working_game_state.player_in_action].select_move(
                game)
            if players[0].id == game.working_game_state.player_in_action:
                players[1].mcts_tree.go_down(game, move)
            else:
                players[0].mcts_tree.go_down(game, move)

            game.apply_move(move)

        game.working_game_state.board.print_board()

        winner = game.final_winner

        score = 0
        if winner == az_agent.id:
            score = 1

        pipe.send(score)
        pipe.close()

    def _evaluate_ploicy_in_parallel(self):

        final_score = 0

        num_of_devices = len(self._devices_ids)

        for _ in range(0, self._evaluate_number_of_games, num_of_devices):
            processes = []
            pipes = []

            for gpu_index in range(num_of_devices):
                parent_connection_end, child_connection_end = mp.Pipe()

                p = mp.Process(target=PolicyEvaluator._evaluate_policy_once_in_parallel, args=(self._basic_mcts_rounds_per_move, self._basic_mcts_c_puct, self._az_mcts_temperature,
                                                                                               self._encoder, self._model, self._az_mcts_rounds_per_move, self._c_puct, self._devices[gpu_index], self._board_size, self._number_of_planes, child_connection_end))

                processes.append(p)
                pipes.append((parent_connection_end, child_connection_end))
                p.start()

            for (parent_connection_end, child_connection_end) in pipes:

                child_connection_end.close()

                while True:
                    try:
                        score = parent_connection_end.recv()
                        self._logger.debug("current score is {}".format(score))

                        final_score += score

                    except EOFError:
                        break

            for (parent_connection_end, _) in pipes:
                parent_connection_end.close()

            for p in tqdm(processes):
                p.join()

        return final_score
