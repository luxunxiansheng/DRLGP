import copy

import math
import random

from tqdm import tqdm

from common.gamestate import GameState
from common.move import Move
from common.player import Player
from agent.randomagent import RandomAgent
from game.tictactoegame import TicTacToeGame


class MCTSNode(object):

    DRAW = -1

    def __init__(self, game, game_state, parent=None, previous_move=None):
        self._game = game
        self._game_state = game_state
        self._parent = parent
        self._previous_move = previous_move
        self._win_counts = {
            game.players[0].id: 0,
            game.players[1].id: 0,
            MCTSNode.DRAW:    0,
        }

        self._num_rollouts = 0
        self._children = []
        self._unvisited_points = game_state.board.get_legal_points()

    @property
    def game_state(self):
        return self._game_state

    @property
    def previous_move(self):
        return self._previous_move

    @property
    def num_rollouts(self):
        return self._num_rollouts

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    def add_random_child(self):
        index = random.randint(0, len(self._unvisited_points)-1)
        new_point = self._unvisited_points.pop(index)
        new_game_state = self._game.transit(self._game_state, Move(new_point))
        new_node = MCTSNode(self._game, new_game_state, self, new_point)
        self._children.append(new_node)
        return new_node

    def record_win(self, winner):
        if winner is not None:
            self._win_counts[winner.id] += 1
        else:
            self._win_counts[MCTSNode.DRAW] += 1
        self._num_rollouts += 1

    def can_add_child(self):
        return len(self._unvisited_points) > 0

    def is_terminal(self):
        return self._game.is_final_state(self._game_state)

    def win_ratio(self, player):
        return float(self._win_counts[player.id]/float(self._num_rollouts))


class MCTSAgent(Player):
    def __init__(self, id, name, num_rounds, temperature):
        super().__init__(self, id, name, mark)
        self._num_rounds = num_rounds
        self._temperature = temperature

    def _select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -10
        best_child = None

        for child in node.children:
            win_ratio = child.win_ratio(node.game_state.player_in_action)
            exploration_factor = math.sqrt(log_rollouts/child.num_rollouts)
            uct_score = win_ratio+self._temperature*exploration_factor

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    def select_move(self, game, game_state):
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

        root = MCTSNode(game, game_state)

        for _ in tqdm(range(self._num_rounds)):
            node = root

            # select: based on a UCT policy
            while(not node.can_add_child() and (not node.is_terminal())):
                node = self._select_child(node)

            # expand: addunvisited child at random
            if node.can_add_child():
                node = node.add_random_child()

            # simulate: random rollout policy
            winner = self.simulate_random_game_for_state(node.game_state)

            # backpropagate
            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_point = None
        best_win_ratio = -1.0

        for child in root.children:
            child_win_ratio = child.win_ratio(game_state.player_in_action)
            if child_win_ratio > best_win_ratio:
                best_win_ratio = child_win_ratio
                best_point = child.previous_point

        return Move(best_point)

    def simulate_random_game_for_state(self,game_state):
        bots = [RandomAgent(0, "RandomAgent0", "X"),
                RandomAgent(1, "RandomAgent1", "O")]

        # current board status
        board = copy.deepcopy(game_state.board)

        # whose 's turn
        player_in_action = game_state.player_in_action

        if player_in_action == bots[0]:
            start_player = bots[0]
        else:
            start_player = bots[1]

        game = TicTacToeGame(board, bots, start_player)

        while not game.is_over():
            move = game.working_game_state.player_in_action.select_move(game, game.working_game_state)
            game.apply_move(move)
            # game.working_game_state.board.print_board()
        winner = game.get_winner(game.working_game_state)

        return winner
