from six.moves import input

from common.player import Player
from common.point import Point
from common.move import Move
from common.board import Board
from game.tictactoegame import TicTacToeGame


def point_from_coords(text):
    col_name = text[0]
    row = int(text[1])

    return Point(row, Board.get_column_indicator_index(col_name)+1)


def main():
    human_player_X = Player(0, "john", "X")
    human_player_O = Player(1, "jack", "O")

    players=[human_player_X, human_player_O]

    game = TicTacToeGame(3,players,human_player_X)

    game.board.print_board()

    while not game.is_over():
        human_move = input('--')
        point = point_from_coords(human_move.strip())
        move = Move(point)
        game.apply_move(move)
        game.gamestate.board.print_board()
    

    winner = game.winner(players)

    if winner is None:
        print("Draw!")
    else:
        print('Winner :' + str(winner.name))


if __name__ == '__main__':
    main()
