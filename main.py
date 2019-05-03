import torch

from tqdm import tqdm


from agent.alphabetaagent import AlphaBetaAgent
from agent.feedforwardnnagent import FeedForwardNeuralNetworkAgent
from agent.humanplayer import HumanPlayer
from agent.mctsagent import MCTSAgent
from agent.minimaxagent import MinmaxAgent
from agent.randomagent import RandomAgent
from common.board import Board
from common.move import Move
from common.point import Point
from common.oneplaneencoder import OnePlaneEncoder
from game.tictactoegame import TicTacToeGame


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    board_size = 5
    total_games = 100

    player_1 = RandomAgent(0, "RandomAgentX", "X")
    player_2 = MCTSAgent(1, "MCTSAgentO", "O", 100, 0.5)

    players = [player_1, player_2]

    start_player = players[0]

    win_counts = {
        players[0].name: 0,
        players[1].name: 0,
        "Draw":          0,
    }

    for _ in tqdm(range(0, total_games)):
        winner = TicTacToeGame.simulate(board_size, players, start_player)
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
