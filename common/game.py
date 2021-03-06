# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /


from collections import deque

from common.board import Board
from common.gamestate import GameState
from common.piece import Piece


class GameStateMemory:
    def __init__(self, capacity):
        self._capacity = capacity
        self._game_states = deque()

    def push(self, game_state):
        self._game_states.append(game_state.board)
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

    def __init__(self, init_game_state, playerlist, state_cache_size=10, is_self_play=False):
        self._players = playerlist
        self._working_game_state = init_game_state
        self._final_winner = None
        self._is_selfplay = is_self_play
        self._state_cache = GameStateMemory(state_cache_size)
        self._state_cache.push(self._working_game_state)

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
        self._working_game_state = self.look_ahead_next_move(
            self._working_game_state, move)
        self._state_cache.push(self._working_game_state)

    def look_ahead_next_move(self, game_state, move):
        piece = Piece(game_state.player_in_action, move.point)
        new_board = Board(game_state.board.board_size, game_state.board.grid)
        new_board.place_piece(piece)
        return GameState(new_board, self.get_player_after_move(game_state.player_in_action), move)

    def get_player_after_move(self, player_in_action):
        pass

    @property
    def final_winner(self):
        return self._final_winner

    @staticmethod
    def winner(board, players):
        pass

    @staticmethod
    def run_episode(board_size, players, start_player):
        pass
