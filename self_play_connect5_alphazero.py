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
from common.board import Board
from game.connect5game import Connect5Game
from models.connect5network import Connect5Network


def main():

    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    the_device = torch.device('cuda' if use_cuda else 'cpu')
   
    number_of_planes = 10
    board_size   =  9 
    round_per_moves =500
   
    encoder = MultiplePlaneEncoder(number_of_planes,board_size)

    model = Connect5Network(number_of_planes, board_size * board_size)
    model.load_state_dict(torch.load('./archived_model/old/1.pth'))
    model.eval()
       
    experience_collector_1 = AlphaZeroExperienceCollector()
    experience_collector_2 = AlphaZeroExperienceCollector()

    agent_1 = AlphaZeroAgent(1,"Agent1","O",encoder,model,round_per_moves,experience_collector_1,device=the_device)
    agent_2 = AlphaZeroAgent(2,"Agent2","X",encoder,model,round_per_moves,experience_collector_2,device=the_device )

    number_of_games = 2000    

    players = [agent_1,agent_2]

    for game_index in tqdm(range(number_of_games)):
        experience_collector_1.reset_episode()
        experience_collector_2.reset_episode()        
        agent_1.reset_memory()
        agent_2.reset_memory()
        
        winner=Connect5Game.run_episode(board_size,players,players[0 if game_index%2== 0 else 1])

        if winner == players[0]:
           players[0].experience_collector.complete_episode(reward=1)
           players[1].experience_collector.complete_episode(reward=-1) 
        if winner == players[1]:
           players[1].experience_collector.complete_episode(reward=1)
           players[0].experience_collector.complete_episode(reward=-1) 

    combined_experiences= AlphaZeroExpericenceBuffer.combine_experience([experience_collector_1,experience_collector_2])
    
    combined_experiences.serialize('./connect5data/1.pth')
  

if __name__ == '__main__':
    main()
