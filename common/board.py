# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /


import copy
import os

from common.piece import Piece
from common.point import Point

alphabet = list(map(chr, range(65, 91)))

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

    

    def __init__(self, board_size):
        self._board_size = board_size
        self._column_indicator = ['  %s' % alphabet[i] for i in range(0, board_size)]
        self._rows = tuple(range(1, board_size+1))
        self._cols = tuple(range(1, board_size+1))

        self._grid = {}


    
    def clone(self):
        clone = Board(self._board_size)
        for key, value in self._grid.items():
            clone._grid[key]=value
        return clone

    @classmethod
    def get_column_indicator_index(cls, char):
        return alphabet.index(char)

    @classmethod
    def get_column_indicator(cls, index):
        return alphabet[index]

    @property
    def board_size(self):
        return self._board_size

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    def place_piece(self, piece):
        assert self.point_is_on_grid(piece.point)
        assert self.get_piece_at_point(piece.point) is None
        self._grid[piece.point] = piece

    def point_is_on_grid(self, point):
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

        print(' '.join(self._column_indicator))

        for row in range(1, self._board_size+1):
            pieces = []
            for col in range(1, self._board_size+1):
                piece = self.get_piece_at_point(Point(row, col))
                pieces.append(str(piece.owner_id)) if piece is not None else pieces.append(' ')
            print('%d %s' % (row, ' | '.join(pieces)))
