from tqdm import tqdm

from agent.alphabetaagent import AlphaBetaAgent
from agent.humanplayer import HumanPlayer
from agent.mctsagent import MCTSAgent
from agent.minimaxagent import MinmaxAgent
from agent.randomagent import RandomAgent
from common.board import Board
from common.move import Move
from common.point import Point
from game.tictactoe.tictactoegame import TicTacToeGame


def experiment(players, start_player):
    game = TicTacToeGame(3, players, start_player)
    while not game.is_over():
        move = game.working_game_state.player_in_action.select_move(
            game, game.working_game_state)
        game.apply_move(move)
        # game.working_game_state.board.print_board()

    winner = game.get_winner(game.working_game_state)
    return winner


def main():

    total_games = 500

    players = [AlphaBetaAgent(0, "AlphaBetaAgentX",    "X"),
               MCTSAgent(  1, "MCTSAgentO",      "O" , 100, 0.5)]

    start_player = players[1]

    win_counts = {
        players[0].name: 0,
        players[1].name: 0,
        "Draw":          0,

    }

    for _ in tqdm(range(0, total_games)):
        winner = experiment(players, start_player)
        if winner is not None:
            win_counts[winner.name] += 1
        else:
            win_counts["Draw"] += 1

    print("************************************************************************************")
    print("{} plays fisrt".format(start_player.name))
    print("------------------------------------------------------------------------------------")
    print("{} win ratio is {:.2f}".format(players[0].name, float(
        win_counts[players[0].name])/float(total_games)))
    print("------------------------------------------------------------------------------------")
    print("{} win ratio is {:.2f}".format(players[1].name, float(
        win_counts[players[1].name])/float(total_games)))
    print("------------------------------------------------------------------------------------")
    print("Draw ratio is {:.2f}".format(float(
        win_counts["Draw"])/float(total_games)))
    print("************************************************************************************")


if __name__ == '__main__':
    main()
