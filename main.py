
from common.board import Board
from common.move import Move
from common.point import Point
from game.tictactoe.agent.alphabetaagent import AlphaBetaAgent
from game.tictactoe.agent.minimaxagent import MinmaxAgent
from game.tictactoe.agent.randomagent import RandomAgent
from game.tictactoe.humanplayer import HumanPlayer
from game.tictactoe.tictactoegame import TicTacToeGame


def experiment(players):
    game = TicTacToeGame(3, players, players[0])
    game.working_game_state.board.print_board()

    while not game.is_over():
        move = game.working_game_state.player_in_action.select_move(
            game, game.working_game_state)
        game.apply_move(move)
        game.working_game_state.board.print_board()

    winner = game.get_winner(game.working_game_state)

    if winner is None:
        print("Draw!")
    else:
        print('Winner is:' + str(winner))

    return winner


def main():
    players = [RandomAgent(   0, "RandomAgent",    "X"), 
               AlphaBetaAgent(1, "AlphaBetaAgent", "O")]

    win_counts = {
        players[0].name: 0,
        players[1].name: 0,

    }

    total_game = 50
    for _ in range(0, total_game):
        winner = experiment(players)
        if winner is not None:
            win_counts[winner] += 1

    print("{} win ratio is {:.2f}".format(players[0].name, float(
        win_counts[players[0].name])/float(total_game)))
    print("{} win ratio is {:.2f}".format(players[1].name, float(
        win_counts[players[1].name])/float(total_game)))


if __name__ == '__main__':
    main()
