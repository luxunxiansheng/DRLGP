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
from common.utils import Utils
from game.connect5game import Connect5Game


def collect_data(agent_1, agent_2, board_size, players, game_index, experience_buffer):
    agent_1.reset()
    agent_2.reset()
    
    winner = Connect5Game.run_episode(board_size, players, players[0 if game_index % 2 == 0 else 1], True)

    if winner == players[0]:
        players[0].experience_collector.complete_episode(reward=1)
        players[1].experience_collector.complete_episode(reward=-1)
    if winner == players[1]:
        players[1].experience_collector.complete_episode(reward=1)
        players[0].experience_collector.complete_episode(reward=-1)

    experience_buffer.combine_experience([agent_1.experience_collector, agent_2.experience_collector])


def improve_policy(experience, game_index, model, optimizer, batch_size, epochs, kl_threshold,device, writer):
      
    
    model.to(device)
    model.train()
    


    batch_data = random.sample(experience.data, batch_size)

    for i in tqdm(range(epochs)):
        states, rewards, visit_counts = zip(*batch_data)

        states = torch.from_numpy(np.array(list(states))).to(device, dtype=torch.float)
        rewards = torch.from_numpy(np.array(list(rewards))).to(device, dtype=torch.float)
        visit_counts = torch.from_numpy(np.array(list(visit_counts))).to(device, dtype=torch.float)
   
        
        visit_sums = visit_counts.sum(dim=1).view((states.shape[0], 1))
        action_policy_target = visit_counts.float() / visit_sums.float()
        value_target = rewards

        [action_policy, value] = model(states)

        log_policy = F.log_softmax(action_policy, dim=1)
        loss_policy = - log_policy * action_policy_target
        loss_policy = loss_policy.sum(dim=1).mean()

        loss_value = F.mse_loss(value.squeeze(), value_target)

        loss = loss_policy + loss_value
        
        entroy = -torch.mean(torch.sum(log_policy*torch.exp(log_policy),dim=1))

        writer.add_scalar('loss', loss.item(), game_index)
        writer.add_scalar('loss_value', loss_value.item(), game_index)
        writer.add_scalar('loss_policy', loss_policy.item(), game_index)
        writer.add_scalar('entroy', entroy.item(), game_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        [updated_action_policy, updated_value] = model(states)
        kl = np.mean(np.sum(action_policy * (np.log(action_policy + 1e-10)-np.log(updated_action_policy+1e-10)),axis=1))
        
        if kl > kl_threshold * 4:
            break 
        
        
def main():
    cfg = Utils.config()

    use_cuda = torch.cuda.is_available()
    the_device = torch.device('cuda' if use_cuda else 'cpu')

    number_of_planes = cfg['GAME'].getint('number_of_planes')
    board_size = cfg['GAME'].getint('board_size')

    number_of_games = cfg['TRAIN'].getint('number_of_games')
    buffer_size = cfg['TRAIN'].getint('buffer_size')
    learning_rate = cfg['TRAIN'].getfloat('learning_rate')
    batch_size = cfg['TRAIN'].getint('batch_size')
    momentum_ = cfg['TRAIN'].getfloat('momentum')
    epochs = cfg['TRAIN'].getint('epochs')
    kl_threshold = cfg['TRAIN'].getfloat('kl_threshold')

    round_per_moves = cfg['MCTS'].getint('round_per_moves')

    encoder = MultiplePlaneEncoder(number_of_planes, board_size)

    writer = SummaryWriter()

    input_shape = (number_of_planes, board_size, board_size)
    model = Connect5Network(input_shape, board_size * board_size)

    experience_collector_1 = AlphaZeroExperienceCollector()
    experience_collector_2 = AlphaZeroExperienceCollector()

    agent_1 = AlphaZeroAgent(1, "Agent1", "O", encoder, model, round_per_moves, experience_collector_1, device=the_device)
    agent_2 = AlphaZeroAgent(2, "Agent2", "X", encoder, model, round_per_moves, experience_collector_2, device=the_device)
    players = [agent_1, agent_2]

    experience_buffer = AlphaZeroExpericenceBuffer(buffer_size)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_)

    for game_index in tqdm(range(1, number_of_games)):

        # collect data via self-playing
        collect_data(agent_1, agent_2, board_size, players, game_index, experience_buffer)

        if experience_buffer.size > batch_size:
           # update the policy with SGD
           improve_policy(experience_buffer, game_index, model, optimizer, batch_size, epochs,kl_threshold,the_device, writer)


if __name__ == '__main__':
    main()
