
import copy
import os

from common.board import Board
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from common.point import Point
from common.piece import Piece


class Game:
    """
    A abstract class about Game. Essentially, it contains a player list and a current working state. 
    """
      
    
    def __init__(self, board, playerlist, start_player):
        self._players = playerlist
        self._board = board
        self._working_game_state = GameState(board, start_player, None)

    
    @property
    def board(self):
        return self._board

    @property
    def working_game_state(self):
        return self._working_game_state

    @property
    def players(self):
        return self._players

    @players.setter
    def players(self, playerlist):
        self._players = playerlist

    @working_game_state.setter
    def working_game_state(self, game_state):
        self._working_game_state = game_state

    def is_final_state(self, game_state):
        pass

    def is_over(self):
        pass

    def get_winner(self, game_state):
        pass

    def apply_move(self, move):
        pass

    def transit(self, game_state, move):
        pass

    def get_player_after_move(self, the_player):
        pass
    
    @staticmethod
    def winner(board, players):
        pass

    @staticmethod
    def simulate(board_size,players,start_player):
        pass
    
