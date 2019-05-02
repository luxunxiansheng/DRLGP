import copy
import os

from common.point import Point
from common.piece import Piece


class Board:
    """
    A chess borad ,which looks like this:

       A  B   C   D  E   F ......
    1
    2     *
    3
    4
    ...

    The column index is represented by letters and the row index is represented by a number.
    B2, for example, stands for the point (column B , row 2)

    A board consists of  board_size*board_size points.

    A point could be occupied by a piece which belongs to one of the players 

    """
 
    alphabet = list(map(chr, range(65, 91)))

    def __init__(self, board_size):
        self._board_size = board_size
        self._column_indicator = ['  %s' % Board.alphabet[i]
                                  for i in range(0, board_size)]
        self._rows = tuple(range(1, board_size+1))
        self._cols = tuple(range(1, board_size+1))

        self._grid = {}

    @classmethod
    def get_column_indicator_index(cls, char):
        return Board.alphabet.index(char)

    @property
    def board_size(self):
        return self._board_size

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    def place_piece(self,piece):
        assert self.point_is_on_grid(piece.point)
        assert self.get_piece_at_point(piece) is None
        self._grid[piece.point] = piece    

    def point_is_on_grid(self,point):
        return 1 <= point.row <= self._board_size and 1 <= point.col <= self._board_size

    def get_piece_at_point(self, point):
        return self._grid.get(point)
  

    def get_legal_points(self):
        points = []
        for row in self._rows:
            for col in self._cols:
                possible_point = Point(row, col)
                if self.is_free_point(possible_point):
                    points.append(possible_point)
        return points

    def is_free_point(self, point):
        return (self.get_piece_at_point(point) is None)

    def print_board(self):
        print('**************************************************')

        print(''.join(self._column_indicator))

        for row in range(1, self._board_size+1):
            pieces = []
            for col in range(1, self._board_size+1):
                piece = self.get_piece_at_point(Point(row, col))
                pieces.append(piece.owner) if piece is not None else pieces.append('')
            print('%d %s' % (row, ' | '.join(pieces)))
