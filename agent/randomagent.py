import random

from common.move import Move
from common.player import Player
from common.point import Point


class RandomAgent(Player):
    def select_move(self,game):
        possible_points = game.working_game_state.board.get_legal_points()
        if not possible_points:
            return None
        return Move(random.choice(possible_points))
