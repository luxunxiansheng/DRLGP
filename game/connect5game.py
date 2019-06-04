import copy

from common.board import Board
from common.game import Game, GameState
from common.move import Move
from common.player import Player
from common.point import Point
from common.piece import Piece


class Connect5Game(Game):
    @staticmethod
    def _connect_5_into_a_line(board, player):
        for row in range(1,board.board_size+1-4):
            for col in range(1,board.board_size):
                if board.get_piece_at_point(Point(row, col)) is not None and board.get_piece_at_point(Point(row, col)).owner == player and board.get_piece_at_point(Point(row+1, col)) is not None and board.get_piece_at_point(Point(row+1, col)).owner == player and board.get_piece_at_point(Point(row+2, col)) is not None and board.get_piece_at_point(Point(row+2, col)).owner == player and board.get_piece_at_point(Point(row+3, col)) is not None and board.get_piece_at_point(Point(row+3, col)).owner == player and board.get_piece_at_point(Point(row+4, col)) is not None and board.get_piece_at_point(Point(row+4, col)).owner == player:
                       return True

        # check horizontal win
        for row in range(1,board.board_size):
            for col in range(1,board.board_size+1-4):
                if board.get_piece_at_point(Point(row, col)) is not None and board.get_piece_at_point(Point(row, col)).owner == player and board.get_piece_at_point(Point(row, col+1)) is not None and board.get_piece_at_point(Point(row, col+1)).owner == player and board.get_piece_at_point(Point(row, col+2)) is not None and board.get_piece_at_point(Point(row, col+2)).owner == player and board.get_piece_at_point(Point(row, col+3)) is not None and board.get_piece_at_point(Point(row, col+3)).owner == player and board.get_piece_at_point(Point(row, col+4)) is not None and board.get_piece_at_point(Point(row, col+4)).owner == player: 
                    return True

        # check / diagonal win
        for row in range(4,board.board_size+1):
            for col in range(1,board.board_size+1-4):
                if board.get_piece_at_point(Point(row, col)) is not None and board.get_piece_at_point(Point(row, col)).owner == player and board.get_piece_at_point(Point(row-1, col+1)) is not None and board.get_piece_at_point(Point(row-1, col+1)).owner == player and board.get_piece_at_point(Point(row-2, col+2)) is not None and board.get_piece_at_point(Point(row-2, col+2)).owner == player and board.get_piece_at_point(Point(row-3, col+3)) is not None and board.get_piece_at_point(Point(row-3, col+3)).owner == player and board.get_piece_at_point(Point(row-4, col+4)) is not None and board.get_piece_at_point(Point(row-4, col+4)).owner == player: 
                    return True  

		# check \ diagnoal win 
        for row in range(1,board.board_size+1-4):
            for col in range(1,board.board_size+1-4):
                if board.get_piece_at_point(Point(row, col)) is not None and board.get_piece_at_point(Point(row, col)).owner == player and board.get_piece_at_point(Point(row+1, col+1)) is not None and board.get_piece_at_point(Point(row+1, col+1)).owner == player and board.get_piece_at_point(Point(row+2, col+2)) is not None and board.get_piece_at_point(Point(row+2, col+2)).owner == player and board.get_piece_at_point(Point(row+3, col+3)) is not None and board.get_piece_at_point(Point(row+3, col+3)).owner == player and board.get_piece_at_point(Point(row+4, col+4)) is not None and board.get_piece_at_point(Point(row+4, col+4)).owner == player: 
                    return True  
        return False

	
    def is_final_state(self,game_state):
        if Connect5Game._connect_5_into_a_line(game_state.board,self._players[0]) or Connect5Game._connect_5_into_a_line(game_state.board,self._players[1]):
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
        return Connect5Game.winner(game_state.board, self._players)


    @staticmethod
    def winner(board, players):
        if Connect5Game._connect_5_into_a_line(board, players[0]):
            return players[0]

        if Connect5Game._connect_5_into_a_line(board, players[1]):
            return players[1]

        return None


    @staticmethod
    def run_episode(board_size, players, start_player):
        board = Board(board_size)
        game = Connect5Game(board, players, start_player)
        while not game.is_over():
            move = game.working_game_state.player_in_action.select_move(game, game.working_game_state)
            game.apply_move(move)
            game.working_game_state.board.print_board()
        
        winner = game.get_winner(game.working_game_state)
        return winner
