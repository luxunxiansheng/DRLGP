
import copy
import os

from common.board import Board
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from common.point import Point
from common.piece import Piece


class Game:
    """
    A abstract class about Game. Essentially, it contains a player list and a current working state. 
    """
    def __init__(self, board, playerlist, start_player):
        self._players = playerlist
        self._working_game_state = GameState(board,start_player, None)
    
    def reset(self,board,playerlist,start_player):
        self._players=playerlist
        self._working_game_state = GameState(board,start_player,None)

    @property
    def working_game_state(self):
        return self._working_game_state

    @property
    def players(self):
        return self._players

    @players.setter
    def players(self, playerlist):
        self._players = playerlist

    @working_game_state.setter
    def working_game_state(self, game_state):
        self._working_game_state = game_state

    def is_final_state(self, game_state):
        pass

    def is_over(self):
        return self.is_final_state(self._working_game_state)

    def get_winner(self, game_state):
        pass

    def apply_move(self, move):
        self._working_game_state = self.transit(self._working_game_state, move)

    def transit(self, game_state, move):
        new_board = copy.deepcopy(game_state.board)
        piece= Piece(game_state.player_in_action,move.point)
        new_board.place_piece(piece)
        return GameState(new_board,self.get_player_after_move(game_state.player_in_action), move)

    def get_player_after_move(self, the_player):
        pass
    
    @staticmethod
    def winner(board, players):
        pass

    @staticmethod
    def run_episode(board_size,players,start_player):
        pass
    
