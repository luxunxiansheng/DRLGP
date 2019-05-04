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
       
       
       
       
        for col in board.cols:
            for row in range(1, board.rows-4):
                for index in range(5):
                    if board.get_piece_at_point(Point(row+index, col)) is None or board.get_piece_at_point(Point(row+index, col)).owner != player:
                        break
            return True      
   
        for row in board.rows:
            for col in range(1, board.cols - 4):
                for index in range(5):
                    if board.get_piece_at_point(Point(row, col+index)) is None or board.get_piece_at_point(Point(row, col+index)).owner != player:
                        break
            
            
            if all(board.get_piece_at_point(Point(row, col)) is not None and board.get_piece_at_point(Point(row, col)).owner == player for col in board.cols):
                return True
        # Diagonal RL to LR
        if all(board.get_piece_at_point(Point(i, i)) is not None and board.get_piece_at_point(Point(i, i)).owner == player for i in range(1, board.board_size+1)):
            return True
        if all(board.get_piece_at_point(Point(i, board.board_size+1-i)) is not None and board.get_piece_at_point(Point(i, board.board_size+1-i)).owner== player for i in range(1, board.board_size+1)):
            return True
        return False    

    @staticmethod
    def _test_row(board,player,row,start_column_index):
        for i in range(5):
            if board.get_piece_at_point(Point(row,start_column_index+i)) is None or board.get_piece_at_point(Point(row + i, start_column_index+i)).owner != player:
                return False
        return True

    @staticmethod
    def _test_column(board,player,column,start_row_index):
        for i in range(5):
            if board.get_piece_at_point(Point(start_row_index+i,column)) is None or board.get_piece_at_point(Point(start_row_index+i,column)).owner != player:
                return False    
        return True

    @staticmethod       
    def _test_diagnoal_from_TL_to_BR(board, player, start_row_index, start__index):
         for i in range(5):
            if board.get_piece_at_point(Point(start_row_index+i,column)) is None or board.get_piece_at_point(Point(start_row_index+i,column)).owner != player:
                return False
        return True

