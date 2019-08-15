import random

from common.move import Move
from common.player import Player
from common.point import Point


class RandomAgent(Player):
    def select_move(self,game):
        possible_points = game.working_game_state.board.get_legal_points()
        possible_moves = [Move(point) for point in possible_points]

        if not possible_moves:
            return None

        return random.choice(possible_moves)
