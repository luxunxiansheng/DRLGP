import copy

from .board import Board
from .gamestate import GameState
from .move import Move
from .player import Player
from .point import Point


class Game:
    def __init__(self, board, playerlist, start_player):
        self._board = board
        self._players = playerlist
        self._next_round_player = start_player
        self._game_state = GameState(board, start_player, None)

    @property
    def board(self):
        return self._board
    
    @property
    def gamestate(self):
        return self._game_state

    @property
    def next_round_player(self):
        return self._next_round_player
 
    @property
    def players(self):
        return self._players

    def is_valid_move(self, move):
        pass

    def is_over(self):
        pass

    def apply_move(self, move):
        pass

    def winner(self):
        pass    

   