import copy
from board import Board
from gamestate import GameState
from move import Move
from player import Player
from point import Point


class Game:
    def __init__(self,board,playerlist,start_player):
        self._board= board
        self._players= playerlist
        self._current_player = start_player

    @property
    def board(self):
        return self._board

    
    
    def apply_move(self, move):
        next_board = copy.deepcopy(self.board)
        next_board.place(self.the_player, move.point)
        return GameState(next_board, self.the_player.other, move)
