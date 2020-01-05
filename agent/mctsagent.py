import copy

import math
import random

from tqdm import tqdm

from common.gamestate import GameState
from common.move import Move
from common.player import Player
from agent.randomagent import RandomAgent


class MCTSNode(object):

    DRAW = -1

    def __init__(self, game, game_state, parent=None):
        self._game = game
        self._game_state = game_state
        self._parent = parent
        self._win_counts = {
            game.players[0].id: 0,
            game.players[1].id: 0,
            MCTSNode.DRAW:    0,
        }

        self._num_rollouts = 0
        self._children = {}
        self._unvisited_points = game_state.board.get_legal_points()

    
    @property
    def game_state(self):
        return self._game_state

    @property
    def num_rollouts(self):
        return self._num_rollouts

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    def get_child(self,point):
        return self._children[point]

    def add_random_child(self):
        index = random.randint(0, len(self._unvisited_points)-1)
        new_point = self._unvisited_points.pop(index)
        new_game_state = self._game.look_ahead_next_move(self._game_state, Move(new_point))
        new_node = MCTSNode(self._game, new_game_state, self)
        self._children[new_point]=new_node
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
    
    
    def go_down(self, point):
        branch = self._working_node.get_child(point)
        self._working_node = branch.child_node

        


class MCTSAgent(Player):
    def __init__(self, id, name,tree,num_rounds, temperature):
        super().__init__(id, name)
        self._num_rounds = num_rounds
        self._temperature = temperature
        self._mcts_tree = tree
    

    @property
    def msct_tree(self):
        return self._mcts_tree

    def _select_child(self, node):
        total_rollouts = sum(child.num_rollouts for _,child in node.children.items())
        log_rollouts = math.log(total_rollouts)

        best_score = -10
        best_child = None

        for _, child in node.children.items():
            win_ratio = child.win_ratio(node.game_state.player_in_action)
            exploration_factor = math.sqrt(log_rollouts/child.num_rollouts)
            uct_score = win_ratio+self._temperature*exploration_factor

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    def select_move(self,game):
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
            self._mcts_tree.working_node= MCTSNode(game, game.working_game_state)

        working_root= self._mcts_tree.working_node

        for _ in tqdm(range(self._num_rounds)):
            node = working_root

            # select: based on a UCT policy
            while(not node.can_add_child() and (not node.is_terminal())):
                node = self._select_child(node)

            # expand: addunvisited child at random
            if node.can_add_child():
                node = node.add_random_child()

            # simulate: random rollout policy
            winner = self._simulate_random_game_for_state(game,node.game_state)

            # backpropagate
            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_point = None
        best_win_ratio = -1.0

        for _,child in working_root.children.items():
            child_win_ratio = child.win_ratio(game.working_game_state.player_in_action)
            if child_win_ratio > best_win_ratio:
                best_win_ratio = child_win_ratio
                best_point = child.previous_point
        
        self._mcts_tree.go_down(best_point)

        return Move(best_point)

    def _simulate_random_game_for_state(self, game,game_state):
        bots = [RandomAgent(game.players[0].id, "RandomAgent0"),
                RandomAgent(game.players[1].id, "RandomAgent1")]

        
        # current board status
        board = copy.deepcopy(game_state.board)

        # whose 's turn
        player_in_action = game_state.player_in_action

        if player_in_action.id == bots[0].id:
            start_player = bots[0]
        else:
            start_player = bots[1]
        
        game = copy.deepcopy(game)
        game.reset(board, bots, start_player)

        while not game.is_over():
            move = game.working_game_state.player_in_action.select_move(game)
            game.apply_move(move)
            #game.working_game_state.board.print_board()
        winner = game.get_winner(game.working_game_state)

        return winner

   
        