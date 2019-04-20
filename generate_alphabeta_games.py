import argparse

import numpy as np
from agent.alphabetaagent import AlphaBetaAgent
from agent.humanplayer import HumanPlayer
from agent.mctsagent import MCTSAgent
from agent.minimaxagent import MinmaxAgent
from agent.randomagent import RandomAgent
from common.board import Board
from common.move import Move
from common.oneplaneencoder import OnePlaneEncoder
from common.point import Point
from game.tictactoe.tictactoegame import TicTacToeGame
from tqdm import tqdm


def experiment(players, start_player):
    
    boards= []
    moves = []
                
    game = TicTacToeGame(3, players, start_player)
    encoder = OnePlaneEncoder(game.working_game_state.board)
    
    while not game.is_over():
        move = game.working_game_state.player_in_action.select_move(
            game, game.working_game_state)

        board_matrix= encoder.encode(game._working_game_state)
        boards.append(board_matrix)

        move_one_hot = np.zeros(encoder.num_points())
        move_one_hot[encoder.encode_point(move.point)] =1
        moves.append(move_one_hot)   

        game.apply_move(move)
        
        # game.working_game_state.board.print_board()
    winner = game.get_winner(game.working_game_state)
    return winner,np.array(boards), np.array(moves)


def load_generated_data():
    boards = np.load('./generated_data/features.npy')
    moves  = np.load('./generated_data/labels.npy')

    return boards, moves


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-games','-n',type=int, default=20)
    

    args= parser.parse_args()

    total_games = args.num_games

    players = [AlphaBetaAgent(0, "AlphaBetaAgentX","X"),
               AlphaBetaAgent(1, "AlphaBetaAgentO","O")]

    start_player = players[1]

    win_counts = {
        players[0].name: 0,
        players[1].name: 0,
        "Draw":          0,
 
    }

    features=[]
    labels=[]

    for _ in tqdm(range(0, total_games)):
        winner,x,y = experiment(players, start_player)
        features.append(x)
        labels.append(y)
        
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

    np.save('./generated_data/features.npy',np.concatenate(features))
    np.save('./generated_data/labels.npy',np.concatenate(labels))  


if __name__ == '__main__':
    main()
