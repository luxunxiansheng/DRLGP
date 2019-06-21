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
    model = Connect5Network(number_of_planes,board_size*board_size)
   
    combined_experiences=AlphaZeroExpericenceBuffer.deserialize('./connect5data/1.pth')
    
    AlphaZeroAgent.train(combined_experiences,model,0.002,128,the_device)
  

if __name__ == '__main__':
    main()
