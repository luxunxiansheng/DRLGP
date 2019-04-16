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
        self._board = copy.copy(self._board)
        self._board.place(self._next_round_player, move.point)
        self._next_round_player = self.get_other_player(self._next_round_player)
        self._game_state = GameState(self._board, self._next_round_player, move)

    def legal_moves(self):
        moves = []
        for row in self.board._rows:
            for col in self.board._cols:
                move = Move(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        return moves

    def _in_a_line(self, player):
        for col in self.board._cols:
            if all(self.board.get_player(Point(row, col)) == player for row in self.board._rows):
                return True

        for row in self.board._rows:
            if all(self.board.get_player(Point(row, col)) == player for col in self.board._cols):
                return True

        # Diagonal RL to LR
        if all(self.board.get_player(Point(i, i)) == player for i in range(1, self.board._board_size+1)):
            return True

        if all(self.board.get_player(Point(i, self.board._board_size+1-i)) == player for i in range(1, self.board._board_size+1)):
            return True

        return False

    def is_over(self):

        if self._in_a_line(self.players[0]):
            return True

        if self._in_a_line(self.players[1]):
            return True

        if all(self.board.get_player(Point(row, col)) is not None
               for row in self.board._rows
               for col in self.board._cols):
            return True

        return False

    def is_valid_move(self, move):
        return (self.board.get_player(move.point) is None and not self.is_over())

    
    def winner(self,players):
        if self._in_a_line(players[0]):
            return players[0]
        
        if self._in_a_line(players[1]):
            return players[1]
        
        return None
    
    def get_other_player(self,the_player):
        if self._players[0]== the_player:
            return self._players[1]
        else:
            return self._players[0]    


