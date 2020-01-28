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


from common.point import Point


class Connect5Game(Game):

    ASSIGNED_PLAYER_ID_1 = 0
    ASSIGNED_PLAYER_ID_2 = 1

    @staticmethod
    def _check_vertical_win(board, player):
        for row in range(1, board.board_size+1-4):
            for col in range(1, board.board_size+1):
                if board.get_piece_at_point(Point(row, col)).owner_id == player and board.get_piece_at_point(Point(row + 1, col)).owner_id == player and board.get_piece_at_point(Point(row + 2, col)).owner_id == player and board.get_piece_at_point(Point(row + 3, col)).owner_id == player and board.get_piece_at_point(Point(row + 4, col)).owner_id == player:
                    return True
        return False

    @staticmethod
    def _check_horizontal_win(board, player):
        for col in range(1, board.board_size+1-4):
            for row in range(1, board.board_size+1):
                if board.get_piece_at_point(Point(row, col)).owner_id == player and board.get_piece_at_point(Point(row, col + 1)).owner_id == player and board.get_piece_at_point(Point(row, col + 2)).owner_id == player and board.get_piece_at_point(Point(row, col + 3)).owner_id == player and board.get_piece_at_point(Point(row, col + 4)).owner_id == player:
                    return True
        return False

    @staticmethod
    def _check_left_diagonal_win(board, player):
        for row in range(5, board.board_size+1):
            for col in range(1, board.board_size+1-4):
                if board.get_piece_at_point(Point(row, col)).owner_id == player and board.get_piece_at_point(Point(row - 1, col + 1)).owner_id == player and board.get_piece_at_point(Point(row - 2, col + 2)).owner_id == player and board.get_piece_at_point(Point(row - 3, col + 3)).owner_id == player and board.get_piece_at_point(Point(row - 4, col + 4)).owner_id == player:
                    return True
        return False

    @staticmethod
    def _check_right_diagonal_win(board, player):
        for row in range(1, board.board_size+1-4):
            for col in range(1, board.board_size+1-4):
                if board.get_piece_at_point(Point(row, col)).owner_id == player and board.get_piece_at_point(Point(row + 1, col + 1)).owner_id == player and board.get_piece_at_point(Point(row + 2, col + 2)).owner_id == player and board.get_piece_at_point(Point(row + 3, col + 3)).owner_id == player and board.get_piece_at_point(Point(row + 4, col + 4)).owner_id == player:
                    return True
        return False

    @staticmethod
    def _connect_5_into_a_line(board, player):

        if Connect5Game._check_vertical_win(board, player):
            return True

        if Connect5Game._check_horizontal_win(board, player):
            return True

        if Connect5Game._check_left_diagonal_win(board, player):
            return True

        if Connect5Game._check_right_diagonal_win(board, player):
            return True

        return False

    def is_final_state(self, game_state):
        if Connect5Game._connect_5_into_a_line(game_state.board, self._players[0]):
            self._final_winner = self._players[0]
            return True

        if Connect5Game._connect_5_into_a_line(game_state.board, self._players[1]):
            self._final_winner = self._players[1]
            return True

        if len(game_state.board.get_legal_points()) == 0:
            self._final_winner = None
            return True

        return False

    def get_player_after_move(self, player_in_action):
        if self._players[0] == player_in_action:
            return self._players[1]
        else:
            return self._players[0]

    def get_winner(self, game_state):
        return Connect5Game.winner(game_state.board, self._players)

    @staticmethod
    def winner(board, players):
        if Connect5Game._connect_5_into_a_line(board, players[0]):
            return players[0]
        if Connect5Game._connect_5_into_a_line(board, players[1]):
            return players[1]
        return None
