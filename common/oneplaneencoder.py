import numpy as np

from .board import Board
from .encoder import Encoder
from .point import Point


class OnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        self._board_size= board_size
        self._board_width = board_size
        self._board_height = board_size
        self._num_plane = 1

    def name(self):
        return 'OnePlaneEncoder'

    
    @property
    def board_width(self):
        return self._board_width

    @property
    def board_height(self):
        return self._board_height

    def encode(self, game_state):
        board_matrix = np.zeros(self.shape(),dtype=int)
        player_in_action = game_state.player_in_action
        for row in range(self._board_height):
            for col in range(self._board_width):
                point = Point(row+1, col+1)
                piece_owner = game_state.board.get_piece_at_point(point).owner
                if piece_owner is not None:
                    if piece_owner == player_in_action:
                        board_matrix[0, row, col] = 1
                    else:
                        board_matrix[0, row, col] = -1
        return board_matrix

    def shape(self):
        return self._num_plane, self._board_height, self._board_width

    def encode_point(self, point):
        return self._board_width*(point.row-1)+(point.col-1)

    def decode_point_index(self, index):
        row = index // self._board_width
        col = index % self._board_width
        return Point(row=row+1, col=col+1)

    def num_points(self):
        return self._board_width*self._board_height

    


