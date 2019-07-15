
import copy
import os

from collections import deque

from common.board import Board
from common.gamestate import GameState
from common.move import Move
from common.player import Player
from common.point import Point
from common.piece import Piece


class Game_State_Memory:
    def __init__(self, capacity):
        self._capacity = capacity
        self._game_states = deque()

    def push(self, experience):
        self._game_states.append(experience)
        if self.size() > self._capacity:
            self._game_states.popleft()

    def size(self):
        return len(self._game_states)

    @property
    def game_states(self):
        return list(self._game_states)

    def clear(self):
        self._game_states.clear()





class Game:
    """
    A abstract class about Game. Essentially, it contains a player list and a current working state. 
    """
    def __init__(self, board, playerlist, start_player,state_cache_size=10,is_self_play=False):
        self._players = playerlist
        self._working_game_state = GameState(board, start_player, None)
        self._final_winner = None
        self._is_selfplay = is_self_play
        self._state_cache = Game_State_Memory(state_cache_size)
        
    
    def reset(self,board,playerlist,start_player,is_self_play=False):
        self._players=playerlist
        self._working_game_state = GameState(board, start_player, None)
        self._final_winner = None
        self._is_selfplay = is_self_play
        self._state_cache.clear()

    @property
    def state_cache(self):
        return self._state_cache 
    
    @property
    def is_selfplay(self):
        return self._is_selfplay

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
        self._working_game_state = self.look_ahead_next_move(self._working_game_state, move)

    def look_ahead_next_move(self, game_state, move):
        new_board = copy.deepcopy(game_state.board)
        piece= Piece(game_state.player_in_action.id,move.point)
        new_board.place_piece(piece)
        return GameState(new_board,self.get_player_after_move(game_state.player_in_action), move)

    def get_player_after_move(self, the_player):
        pass
    
    @property
    def final_winner(self):
        return self._final_winner

    @staticmethod
    def winner(board, players):
        pass

    @staticmethod
    def run_episode(board_size,players,start_player):
        pass
    
