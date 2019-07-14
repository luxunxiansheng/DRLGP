import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from agent.alphazeroagent import (AlphaZeroAgent, AlphaZeroExpericenceBuffer,
                                  AlphaZeroExperienceCollector,
                                  MultiplePlaneEncoder)

from agent.mctsagent import  MCTSAgent 

from common.board import Board
from game.connect5game import Connect5Game
from models.connect5network import Connect5Network

def evaluate_plicy(board_size):
    mcts_agent_1 = MCTSAgent(0,"MCTSAgent1","O",70,1.4)
    mcts_agent_2 = MCTSAgent(1,"MCTSAgent2","X",10,1.4)
    number_of_games = 10    
    
    win_counts = {
        mcts_agent_1.id: 0,
        mcts_agent_2.id: 0,
    }

    for game_index in tqdm(range(number_of_games)):
        
        players = [mcts_agent_1, mcts_agent_2]
        winner=Connect5Game.run_episode(board_size,players,players[0 if game_index%2== 0 else 1],is_self_play=False)
        
        if winner is not None:
            win_counts[winner.id] += 1
        
    return win_counts[mcts_agent_1.id]/number_of_games

def main():

    print(evaluate_plicy(8))
  
  

if __name__ == '__main__':
    main()
