
from common.board import Board
from common.move import Move
from common.point import Point
from game.tictactoe.agent.minimaxagent import MinmaxAgent
from game.tictactoe.agent.randomagent import RandomBot
from game.tictactoe.humanplayer import HumanPlayer
from game.tictactoe.tictactoegame import TicTacToeGame


def main():
    human_player_X = HumanPlayer(0, "Human", "X")
    random_bot_O = RandomBot(1, "Robot", "O")
    #minimax_bot_O= MinmaxAgent(1,"MinimaxBox","O")

    players = [human_player_X, random_bot_O]

    game = TicTacToeGame(3, players, human_player_X)

    game.working_game_state.board.print_board()

    while not game.is_over():
        move = game.working_game_state.player_in_action.select_move(game.working_game_state)
        game.apply_move(move)
        game.working_game_state.board.print_board()

    winner = game.get_winner(game.working_game_state)

    if winner is None:
        print("Draw!")
    else:
        print('Winner is:' + str(winner.name))


if __name__ == '__main__':
    main()
