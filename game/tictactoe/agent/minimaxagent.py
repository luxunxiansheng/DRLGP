import random

from tqdm import tqdm

from common.move import Move
from common.player import Player
from common.point import Point


class MinmaxAgent(Player):
    def select_move(self, game_state):
        assert game_state.player_in_action == self

        winning_moves = []
        draw_moves = []
        losing_moves = []

        for possible_point in game_state.board.get_legal_points():
            possible_move = Move(possible_point)
            player_after_possible_move = self._game.get_other_player_after_move(self)
            next_state = game_state.transit(possible_move, player_after_possible_move)
            our_best_outcome = self.best_result(next_state, self)

            # We set the player_0 as the  maximizer, the best score is 1.0
            if self == self._game.player[0]:
                if our_best_outcome == 1.0:
                    winning_moves.append(possible_move)
                if our_best_outcome == -1.0:
                    losing_moves.append(possible_move)
                else:
                    draw_moves.append(possible_move)

            # We set the player_0 as the  maximizer, the best score is -1.0
            if self == self._game.player[1]:
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
        assert game_state.player_in_action == the_player

        if self._game.is_final_state(game_state):
            if self._game.get_winner(game_state) == self._game.player[0]:
                return 1.0
            if self._game.get_winner(game_state) == self._game.player[1]:
                return -1.0
            else:
                return 0.0
        # for the maximizer
        if the_player == self._game.player[0]:
            max_value = -2.0
            for possible_point in tqdm(game_state.board.get_legal_points()):
                possible_move=Move(the_player,possible_point)
                player_after_possible_move = self._game.get_other_player_after_move(the_player)
                next_game_state = game_state.transit(possible_move,player_after_possible_move)
                the_value = self.best_result(next_game_state, self._game.player[1])
                max_value = max(max_value, the_value)
            return max_value
        # for the minimizer 
        if the_player == self._game.player[1]:
            min_value = 2.0
            for possible_point in tqdm(game_state.board.get_legal_points()):
                possible_move=Move(the_player,possible_point)
                player_after_possible_move = self._game.get_other_player_after_move(the_player)
                next_game_state = game_state.transit(possible_move,player_after_possible_move)
                the_value = self.best_result(next_game_state, self._game.player[0])
                max_value = min(max_value, the_value)
            return min_value
           
