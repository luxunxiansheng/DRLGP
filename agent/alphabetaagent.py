import random

from common.move import Move
from common.player import Player
from common.point import Point


class AlphaBetaAgent(Player):
    def select_move(self, game, game_state):
        assert game_state.player_in_action== self

        winning_moves = []
        draw_moves = []
        losing_moves = []

        for possible_point in game_state.board.get_legal_points():
            possible_move = Move(possible_point)
            next_state = game.transit(game_state, possible_move)
            our_best_outcome = self.minmax(game,next_state,-2.0,2.0)

            # We set the player_0 as the  maximizer, the best score is 1.0
            if self == game.players[0]:
                if our_best_outcome == 1.0:
                    winning_moves.append(possible_move)
                if our_best_outcome == -1.0:
                    losing_moves.append(possible_move)
                else:
                    draw_moves.append(possible_move)

            # We set the player_0 as the  maximizer, the best score is -1.0
            if self == game.players[1]:
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

    def minmax(self, game, game_state,best_max,best_min):
        if game.is_final_state(game_state):
            if game.get_winner(game_state) == game.players[0]:
                return 1.0
            if game.get_winner(game_state) == game.players[1]:
                return -1.0
            else:
                return 0.0
        # for the maximizer
        if game_state.player_in_action == game.players[0]:
            max_value = -2.0
            for possible_point in game_state.board.get_legal_points():
                possible_move = Move(possible_point)
                next_game_state = game.transit(game_state, possible_move)
                the_value = self.minmax(game, next_game_state,best_max,best_min)
                max_value = max(max_value, the_value)
                best_max  = max(best_max,the_value)
                if best_max >= best_min:
                    break
            return max_value
        # for the minimizer
        if game_state.player_in_action == game.players[1]:
            min_value = 2.0
            for possible_point in game_state.board.get_legal_points():
                possible_move = Move(possible_point)
                next_game_state = game.transit(game_state, possible_move)
                the_value = self.minmax(game, next_game_state,best_max,best_min)
                min_value = min(min_value,the_value)
                best_min =  min(best_min,the_value)
                if best_min <= best_max:
                    break
            return min_value
