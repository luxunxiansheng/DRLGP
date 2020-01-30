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

from tqdm import tqdm

from agent.mcts.mctsnode import MCTSNode
from agent.mcts.mctstree import MCTSTree
from agent.randomagent import RandomAgent
from common.board import Board
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from game.connect5game import Connect5Game


class MCTSAgent(Player):
    def __init__(self, id, name, num_rounds, temperature):
        super().__init__(id, name)
        self._num_rounds = num_rounds
        self._temperature = temperature
        self._mcts_tree = MCTSTree()

    @property
    def mcts_tree(self):
        return self._mcts_tree

    def select_move(self, game):
        # The basic MCTS process is described as below:
        #
        # Selection:
        #
        # Start from root and select successive child nodes until a leaf node is
        # reached. The  root is the current game state and a leaf is any node from which no
        # simulation has yet been initated. The selection will let the game tree expand
        # towards the most promising moves.
        #
        # Expansion:
        # Unless leaf node ends the game decisively for either player, create one(or more)
        # child nodes and choose one of them.
        #
        # Simulation:
        # Complete one random playout from created node.
        #
        # Backpropagation:
        # Use the result of the rollout to update information in the nodes on the path from
        # created node to root
        #

        if self._mcts_tree.working_node is None:
            self._mcts_tree.working_node = MCTSNode(
                game.working_game_state, 1.0, None)

        for _ in range(self._num_rounds):
            node = self._mcts_tree.working_node
            while True:
                if node.is_leaf():
                    break
                # select: based on a UCT policy
                node = node.select(self._temperature)

            leaf_value = 0.0
            if not game.is_final_state(node.game_state):
                free_points = node.game_state.board.get_legal_points()
                prior = 1.0/len(free_points)
                for point in free_points:
                    node.add_child(game, point, prior)
                # simulate: random rollout policy
                leaf_value = self._simulate_random_game_for_state(
                    node.game_state)
            else:
                if game.final_winner is not None:
                    leaf_value = 1.0 if game.final_winner == node.game_state.player_in_action else -1.0

            node.update_recursively(self._mcts_tree.working_node, -leaf_value)

        # self._mcts_tree.working_node.game_state.board.print_visits(self._mcts_tree.working_node.children)

        best_point = max(self._mcts_tree.working_node.children.items(
        ), key=lambda point_node: point_node[1].num_visits)[0]

        self._mcts_tree.go_down(game, Move(best_point))

        return Move(best_point)

    def _simulate_random_game_for_state(self, game_state):
        random_agent_0 = RandomAgent(
            Connect5Game.ASSIGNED_PLAYER_ID_1, "RandomAgent0")
        random_agent_1 = RandomAgent(
            Connect5Game.ASSIGNED_PLAYER_ID_2, "RandomAgent1")

        bots = {}
        bots[random_agent_0.id] = random_agent_0
        bots[random_agent_1.id] = random_agent_1

        # current board status
        board = Board(game_state.board.board_size, game_state.board.grid)
        init_game_state = GameState(board, game_state.player_in_action, None)

        game = Connect5Game(
            init_game_state, [random_agent_0.id, random_agent_1.id], 0)

        # game.working_game_state.board.print_board()

        while not game.is_over():
            move = bots[game.working_game_state.player_in_action].select_move(
                game)
            game.apply_move(move)

        winner = game.final_winner
        # game.working_game_state.board.print_board()

        if winner is None:
            return 0.0
        else:
            return 1.0 if winner == game_state.player_in_action else -1.0
