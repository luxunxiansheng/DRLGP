
import copy
from point import Point
from player import Player
from board import Board
from move import Move


class GameState:
    def __init__(self, board, the_player, last_move):
        self._board = board
        self._the_player = the_player
        self._last_move = last_move

    @property
    def the_player(self):
        return self.the_player

    @property
    def board(self):
        return self._board

    @property
    def last_move(self):
        return self._last_move

    @classmethod
    def new_game(cls, board_size):
        board = Board(board_size)
        return GameState(board, Player.x, None)

    def is_valid_move(self, move):
        return(self.board.get(move.point) is None and not self.is_over())

    def legal_moves(self):
        moves = []
        for row in self.board._rows:
            for col in self.board._cols:
                move = Move(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        return moves

    def is_over(self):

        if self._in_a_row(Player.x):
            return True

        if self._in_a_row(Player.o):
            return True

        if all(self.board.get(Point(row, col)) is not None
               for row in self.board._rows
               for col in self.board._cols):
            return True

        return False

    def _in_a_row(self, player):
        for col in self.board._cols:
            if all(self.board.get(Point(row, col)) == player for row in self.board._rows):
                return True

        for row in self.board._rows:
            if all(self.board.get(Point(row, col)) == player for col in self.board._cols):
                return True

        # Diagonal RL to LR
        if all(self.board.get(Point(i, i)) == player for i in range(1, self.board._board_size+1)):
            return True

        if all(self.board.get(Point(i, self.board._board_size+1-i)) == player for i in range(1, self.board._board_size+1)):
            return True

        return False

    def winner(self):
        if self._in_a_row(Player.x):
            return Player.x

        if self._in_a_row(Player.o):
            return Player.o
        return None
