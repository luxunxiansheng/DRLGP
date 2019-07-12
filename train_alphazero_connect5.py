import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.connect5network import Connect5Network
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent.alphazeroagent import (AlphaZeroAgent, AlphaZeroExpericenceBuffer,
                                  AlphaZeroExperienceCollector,
                                  MultiplePlaneEncoder)
from common.board import Board
from game.connect5game import Connect5Game


def main():

    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    the_device = torch.device('cuda' if use_cuda else 'cpu')

    number_of_planes = 10
    board_size = 9

    input_shape=(number_of_planes,board_size,board_size) 

    model = Connect5Network(input_shape, board_size * board_size)
    #model.load_state_dict(torch.load('./archived_model/old/1.pth'))
    

    combined_experiences = AlphaZeroExpericenceBuffer.deserialize('./connect5data/1000.pth')

    writer = SummaryWriter()

    AlphaZeroAgent.train(combined_experiences, model, 0.0001, 32, the_device, writer)

    torch.save(model.state_dict(), './archived_model/new/1.pth')

    writer.close()


if __name__ == '__main__':
    main()
