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
    round_per_moves =100
   
    encoder = MultiplePlaneEncoder(number_of_planes,board_size)

    input_shape=(number_of_planes,board_size,board_size) 

    model_old = Connect5Network(input_shape, board_size * board_size)
    #model_old.load_state_dict(torch.load('./archived_model/old/1.pth'))
    model_old.eval()



    model_new = Connect5Network(input_shape, board_size * board_size)
    model_new.load_state_dict(torch.load('./archived_model/new/1.pth'))
    model_new.eval()
       
    

    agent_old = AlphaZeroAgent(1,"Agent_Old","O",encoder,model_old,round_per_moves,None,device=the_device)
    agent_new = AlphaZeroAgent(2,"Agent_New","X",encoder,model_new,round_per_moves,None,device=the_device )

    number_of_games = 100    

    players = [agent_old, agent_new]
    

    win_counts = {
        agent_old.id: 0,
        agent_new.id: 0,
       
    }

    for game_index in tqdm(range(number_of_games)):
          
        agent_old.reset_memory()
        agent_new.reset_memory()
        
        winner=Connect5Game.run_episode(board_size,players,players[0 if game_index%2== 0 else 1],is_self_play=False)
        
        if winner is not None:
            win_counts[winner.id] += 1
       
    print('New model:old Model {}:{}'.format(win_counts[agent_new.id],win_counts[agent_old.id]))

    if win_counts[agent_new.id] / number_of_games > 0.6:
       torch.save(model_new.state_dict(), './archived_model/old/1.pth')
      
   
       
      
    
    
  

if __name__ == '__main__':
    main()
