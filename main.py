from common.player import Player
from game.tictactoegame import TicTacToeGame


def main():
    
    human_player_X = Player(0,"john","X")
    human_player_O = Player(1,"jack","O")

    game = TicTacToeGame(3,[human_player_X,human_player_O],human_player_X)

    game.board.print_board()

    """ while not game.is_over():
        if game.the_player == human_player:
            human_move = input('--')
            point = point_from_coords(human_move.strip())
            move = Move(point)
        else:
            move = bot.select_move(game)
        game = game.apply_move(move)

        game.board.print_board()

    winner = game.winner()

    if winner is None:
        print("Draw!")
    else:
        print('Winner :' + str(winner)) """


if __name__ == '__main__':
    main()