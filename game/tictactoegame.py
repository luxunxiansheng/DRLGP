import copy

from common.board import Board
from common.game import Game, GameState
from common.move import Move
from common.player import Player
from common.point import Point


class TicTacToeGame(Game):
    def __init__(self, board_size, playerlist, start_player):
        board = Board(board_size)
        Game.__init__(self, board, playerlist, start_player)

    def apply_move(self, move):
        next_board = copy.deepcopy(self.board)
        next_board.place(self.next_round_player, move.point)
        the_player_in_next_turn = self.get_other_players(
            self.next_round_player)[0]
        self._game_state = GameState(next_board, the_player_in_next_turn, move)

    def legal_moves(self):
        moves = []
        for row in self.board._rows:
            for col in self.board._cols:
                move = Move(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        return moves

    def _in_a_row(self, player):
        for col in self.board._cols:
            if all(self.board.get(Point(row, col)) == player for row in self.board._rows):
                return True

        for row in self.board._rows:
            if all(self.board.get(Point(row, col)) == player for col in self.board._cols):
                return True

        # Diagonal RL to LR
        if all(self.board.get(Point(i, i)) == player for i in range(1, self.board._board_size+1)):
            return True

        if all(self.board.get(Point(i, self.board._board_size+1-i)) == player for i in range(1, self.board._board_size+1)):
            return True

        return False

    def is_over(self):

        if self._in_a_row(self.players[0]):
            return True

        if self._in_a_row(self.players[1]):
            return True

        if all(self.board.get(Point(row, col)) is not None
               for row in self.board._rows
               for col in self.board._cols):
            return True

        return False

    def is_valid_move(self, move):
        return (self.board.get(move.point) is None and not self.is_over())
