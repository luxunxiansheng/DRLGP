import random

from common.player import Player
from common.move  import Move
from common.point import Point


class MinmaxAgent(Player):
    def __init__(self, player):
        self.the_player = player

    def select_move(self,game,game_state):
        winning_moves = []
        draw_moves = []
        losing_moves = []

        for possible_move in game.legal_moves():
            next_state = game_state.apply_move(possible_move)
            our_best_outcome = self.best_result(next_state, self.the_player)

            if self.the_player == Player.x:
                if our_best_outcome == 1.0:
                    winning_moves.append(possible_move)
                if our_best_outcome == -1.0:
                    losing_moves.append(possible_move)
                else:
                    draw_moves.append(possible_move)

            if self.the_player == Player.o:
                if our_best_outcome == -1.0:
                    winning_moves.append(possible_move)
                if our_best_outcome == 1.0:
                    losing_moves.append(possible_move)
                else:
                    draw_moves.append(possible_move)

        if winning_moves:
            return random.choice(winning_moves)

        if draw_moves:
            return random.choice(draw_moves)

        return random.choice(losing_moves)

    def best_result(self, game_state, the_player):
        if game_state.is_over():
            if game_state.winner() == Player.x:
                return 1.0
            if game_state.winner() == Player.o:
                return -1.0
            else:
                return 0.0

        if the_player == Player.x:
            max_value = -2.0
            for candidate_move in tqdm(game_state.legal_moves()):
                candidate_state = game_state.apply_move(candidate_move)
                the_value = self.best_result(candidate_state, Player.o)
                max_value = max(max_value, the_value)
            return max_value

        if the_player == Player.o:
            min_value = 2.0
            for candidate_move in tqdm(game_state.legal_moves()):
                candidate_state = game_state.apply_move(candidate_move)
                the_value = self.best_result(candidate_state, Player.x)
                min_value = min(min_value, the_value)

            return min_value