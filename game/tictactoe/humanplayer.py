from six.moves import input

from common.board import Board
from common.move import Move
from common.player import Player
from common.point import Point


class HumanPlayer(Player):

    @classmethod
    def point_from_coords(cls, text):
        col_name = text[0]
        row = int(text[1])
        return Point(row, Board.get_column_indicator_index(col_name)+1)

    def select_move(self,game,game_state):
        human_move = input('--')
        point = HumanPlayer.point_from_coords(human_move.strip())
        return Move(point)
