import copy

from .board import Board
from .gamestate import GameState
from .move import Move
from .player import Player
from .point import Point


class Game:
    def __init__(self, board, playerlist, start_player):
        self._players = playerlist
        self._working_game_state = GameState(board,start_player, None)
    
    @property
    def working_game_state(self):
        return self._working_game_state

    @property
    def players(self):
        return self._players

    def is_over(self):
        pass

    def get_winner(self):
        pass    

    def apply_move(self,move):
        pass
    


   