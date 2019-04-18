import copy

from common.board import Board
from common.game import Game, GameState
from common.move import Move
from common.player import Player
from common.point import Point


class TicTacToeGame(Game):
    def __init__(self, board_size, playerlist, start_player):
        Game.__init__(self, Board(board_size), playerlist, start_player)
    
    def apply_move(self, move):
        self._working_game_state = self.transit(self._working_game_state, move)

    @staticmethod
    def _connect_into_a_line(board, player):
        for col in board.cols:
            if all(board.get_player(Point(row, col)) == player for row in board.rows):
                return True
        for row in board.rows:
            if all(board.get_player(Point(row, col)) == player for col in board.cols):
                return True
        # Diagonal RL to LR
        if all(board.get_player(Point(i, i)) == player for i in range(1, board.board_size+1)):
            return True
        if all(board.get_player(Point(i, board.board_size+1-i)) == player for i in range(1, board.board_size+1)):
            return True
        return False
    
    def is_final_state(self, game_state):
        if TicTacToeGame._connect_into_a_line(game_state.board, self._players[0]) or TicTacToeGame._connect_into_a_line(game_state.board, self._players[1]):
            return True

        if all(game_state.board.get_player(Point(row, col)) is not None
               for row in game_state.board.rows
               for col in game_state.board.cols):
            return True

        return False

    def get_player_after_move(self, player_in_action):
        if self._players[0]== player_in_action:
            return self._players[1]
        else:
            return self._players[0]

    def is_over(self):
        return self.is_final_state(self._working_game_state)

    def get_winner(self, game_state):
        return TicTacToeGame.winner(game_state.board, self._players)

    def transit(self, game_state, move):
        new_board = copy.deepcopy(game_state.board)
        new_board.place(game_state.player_in_action, move.point)
        return GameState(new_board, self.get_player_after_move(game_state.player_in_action), move)

    @staticmethod
    def winner(board, players):
        if TicTacToeGame._connect_into_a_line(board, players[0]):
            return players[0]

        if TicTacToeGame._connect_into_a_line(board, players[1]):
            return players[1]

        return None
