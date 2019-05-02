
import copy

from common.board import Board
from common.move import Move
from common.player import Player
from common.point import Point


class GameState:
    """
    board: what the situation looks like
    player_in_action: the player who will place its piece into the board
    previous_move:  the move which led to what the current board looks like 
    """
    def __init__(self, board, player_in_action, previous_move):
        self._board = board
        self._player_in_action = player_in_action
        self._previous_move = previous_move

    @property
    def board(self):
        return self._board

    @property
    def player_in_action(self):
        return self._player_in_action

    @property
    def previous_move(self):
        return self._previous_move
