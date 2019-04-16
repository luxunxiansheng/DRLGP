
from common.point import Point
from common.move import Move
from common.board import Board
from game.tictactoe.tictactoegame import TicTacToeGame
from game.tictactoe.humanplayer import HumanPlayer
from game.tictactoe.agent.randomagent import RandomBot

def main():
    human_player_X = HumanPlayer(0, "Human", "X")
    random_bot_O = RandomBot(1, "Robot", "O")

    players=[human_player_X, random_bot_O]

    game = TicTacToeGame(3,players,human_player_X)

    game.board.print_board()

    while not game.is_over():
        move= game.next_round_player.select_move(game,game.game_state)
        game.apply_move(move)
        game.game_state.board.print_board()
   

    winner = game.winner(players)

    if winner is None:
        print("Draw!")
    else:
        print('Winner :' + str(winner.name))


if __name__ == '__main__':
    main()
