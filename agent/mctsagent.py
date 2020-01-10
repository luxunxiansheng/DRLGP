import copy
import math
import random

import numpy as np
from profilehooks import profile
from tqdm import tqdm

from agent.randomagent import RandomAgent
from game.connect5game import Connect5Game
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from common.board import Board



class MCTSNode(object):
    def __init__(self, game_state, prior_p, parent=None):
        self._game_state = game_state
        self._parent = parent
        self._children = {}
        self._num_visits = 0
        
        self._Q = 0
        self._P = prior_p
        self._u = 0

    def select(self,c_puct):
        child=max(self._children.items(),key=lambda point_node: point_node[1].get_value(c_puct))
        return child[1]
    
    def update_recursively(self,root_node,leaf_value):
        if self!=root_node and self._parent:
            self._parent.update_recursively(root_node,-leaf_value)
                
        self._num_visits +=1
        self._Q += 1.0*(leaf_value-self._Q)/self._num_visits

    def is_leaf(self):
        return self._children == {}
    
    @property
    def num_visits(self):
        return self._num_visits

    @property
    def game_state(self):
        return self._game_state

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    def get_child(self,point):
        return self._children.get(point)

    def add_child(self,game,new_point,prior):
        new_game_state = game.look_ahead_next_move(self._game_state, Move(new_point))
        new_node = MCTSNode(new_game_state,prior,self)
        self._children[new_point]=new_node
        return new_node

    def is_terminal(self,game):
        return game.is_final_state(self._game_state)

    def get_value(self,c_puct):
        """
        Same as in alphazero
        """
        self._u = (c_puct * self._P *np.sqrt(self._parent._num_visits) / (1 + self._num_visits))
        return self._Q + self._u


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
    
    def go_down(self,game,move):
        if self._working_node is not None:
            child = self._working_node.get_child(move.point)
            if child is None:
                if not self._working_node.is_terminal(game):
                    child =  self._working_node.add_child(game,move.point,1.0)
            
            self._working_node = child

class MCTSAgent(Player):
    def __init__(self, id, name,num_rounds, temperature):
        super().__init__(id, name)
        self._num_rounds = num_rounds
        self._temperature = temperature
        self._mcts_tree = MCTSTree()
    

    @property
    def mcts_tree(self):
        return self._mcts_tree

    
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
            self._mcts_tree.working_node= MCTSNode(game.working_game_state,1.0,None)
        
        
        for _ in tqdm(range(self._num_rounds)):
            node = self._mcts_tree.working_node
            while True:
                if node.is_leaf():
                    break
                # select: based on a UCT policy
                node = node.select(self._temperature)
            
            if not node.is_terminal(game):
                free_points = node.game_state.board.get_legal_points()
                prior= 1.0/len(free_points)
                for point in free_points:
                    node.add_child(game,point,prior)
            
            # simulate: random rollout policy
            leaf_value= self._simulate_random_game_for_state(node.game_state)
            node.update_recursively(self._mcts_tree.working_node,-leaf_value)

        self._mcts_tree.working_node.game_state.board.print_visits(self._mcts_tree.working_node.children)   

        best_point= max(self._mcts_tree.working_node.children.items(),key=lambda point_node: point_node[1].num_visits)[0]

        self._mcts_tree.go_down(game,Move(best_point))

        return Move(best_point)

    
    def _simulate_random_game_for_state(self,game_state):
        random_agent_0= RandomAgent(Connect5Game.ASSIGNED_PLAYER_ID_1,"RandomAgent0")
        random_agent_1= RandomAgent(Connect5Game.ASSIGNED_PLAYER_ID_2,"RandomAgent1")
        
        bots={}
        bots[random_agent_0.id]=random_agent_0
        bots[random_agent_1.id]=random_agent_1
        
        # current board status
        board = Board(game_state.board.board_size,game_state.board.grid)
        init_game_state= GameState(board,game_state.player_in_action,None)

        game = Connect5Game(init_game_state,[random_agent_0.id,random_agent_1.id],0) 

        #game.working_game_state.board.print_board()

        while not game.is_over():
            move = bots[game.working_game_state.player_in_action].select_move(game)
            game.apply_move(move)
        
        
        winner = game.final_winner
        #game.working_game_state.board.print_board()

        if winner is None:
            return 0
        else:
            return 1 if winner== game_state.player_in_action else -1
