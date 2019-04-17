
import copy

from .board import Board
from .move import Move
from .player import Player
from .point import Point


class GameState:
    def __init__(self, board,next_player, last_move):
        self._board = board
        self._next_round_player = next_player
        self._last_move = last_move

    @property
    def board(self):
        return self._board

    @property
    def next_round_player(self):
        return self._next_round_player

    @property
    def last_move(self):
        return self._last_move


    def transit(self,move,the_nexet_player_after_move):
        new_board = copy.copy(self._board)
        new_board.place(self._next_round_player,move.point)
        return GameState(new_board,the_nexet_player_after_move,move)
