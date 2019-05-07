import copy

from common.board import Board
from common.game import Game, GameState
from common.move import Move
from common.player import Player
from common.point import Point
from common.piece import Piece


class TicTacToeGame(Game):
    @staticmethod
    def _connect_into_a_line(board, player):
        for col in board.cols:
            if all(board.get_piece_at_point(Point(row, col)) is not None and board.get_piece_at_point(Point(row, col)).owner == player for row in board.rows):
                return True
        for row in board.rows:
            if all(board.get_piece_at_point(Point(row, col)) is not None and board.get_piece_at_point(Point(row, col)).owner == player for col in board.cols):
                return True
        # Diagonal RL to LR
        if all(board.get_piece_at_point(Point(i, i)) is not None and board.get_piece_at_point(Point(i, i)).owner == player for i in range(1, board.board_size+1)):
            return True
        if all(board.get_piece_at_point(Point(i, board.board_size+1-i)) is not None and board.get_piece_at_point(Point(i, board.board_size+1-i)).owner == player for i in range(1, board.board_size+1)):
            return True
        return False

    def is_final_state(self, game_state):
        if TicTacToeGame._connect_into_a_line(game_state.board, self._players[0]) or TicTacToeGame._connect_into_a_line(game_state.board, self._players[1]):
            return True

        if all(game_state.board.get_piece_at_point(Point(row, col)) is not None
               for row in game_state.board.rows
               for col in game_state.board.cols):
            return True
        return False

    def get_player_after_move(self, player_in_action):
        if self._players[0] == player_in_action:
            return self._players[1]
        else:
            return self._players[0]

    def get_winner(self, game_state):
        return TicTacToeGame.winner(game_state.board, self._players)

    @staticmethod
    def winner(board, players):
        if TicTacToeGame._connect_into_a_line(board, players[0]):
            return players[0]

        if TicTacToeGame._connect_into_a_line(board, players[1]):
            return players[1]

        return None

    @staticmethod
    def run_episode(board_size, players, start_player):
        board = Board(board_size)
        game = TicTacToeGame(board, players, start_player)
        while not game.is_over():
            move = game.working_game_state.player_in_action.select_move(
                game, game.working_game_state)
            game.apply_move(move)
            # game.working_game_state.board.print_board()

        winner = game.get_winner(game.working_game_state)
        return winner
