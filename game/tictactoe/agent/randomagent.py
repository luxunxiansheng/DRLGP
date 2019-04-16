import random

from common.player import Player
from common.move  import Move
from common.point import Point

class RandomBot(Player):
    def select_move(self,game,game_state):
        possible_moves = []
        for r in range(1, game.board.board_size+1):
            for c in range(1, game.board.board_size+1):
                possible_move = Move(Point(row=r, col=c))
                if game.is_valid_move(possible_move):
                    possible_moves.append(possible_move)

        if not possible_moves:
            return None

        return random.choice(possible_moves)

