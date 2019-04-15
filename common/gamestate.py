
import copy

from .board import Board
from .move import Move
from .player import Player
from .point import Point


class GameState:
    def __init__(self, board, next_player, last_move):
        self._board = board
        self._the_next_round_player = next_player
        self._last_move = last_move

    @property
    def the_next_round_player(self):
        return self._the_next_round_player

    @property
    def board(self):
        return self._board

    @property
    def last_move(self):
        return self._last_move
