
import copy

from .board import Board
from .move import Move
from .player import Player
from .point import Point


class GameState:
    def __init__(self, board,player_in_action, last_move):
        self._board = board
        self._player_in_action = player_in_action
        self._last_move = last_move

    @property
    def board(self):
        return self._board

    @property
    def player_in_action(self):
        return self._player_in_action

    @property
    def last_move(self):
        return self._last_move


    def transit(self,move,the_next_player_after_move):
        new_board = copy.copy(self._board)
        new_board.place(move.player,move.point)
        return GameState(new_board,the_next_player_after_move,move)
