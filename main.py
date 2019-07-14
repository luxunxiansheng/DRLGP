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
from agent.mctsagent import  MCTSAgent  

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

    print('Experience buffer is {}'.format(experience_buffer.size()))


def improve_policy(experience, game_index, model, optimizer, batch_size, epochs, kl_threshold,device, writer):
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
        
        with torch.no_grad():
            entroy = -torch.mean(torch.sum(log_policy*F.softmax(action_policy,dim=1),dim=1))

        writer.add_scalar('loss', loss.item(), game_index)
        writer.add_scalar('loss_value', loss_value.item(), game_index)
        writer.add_scalar('loss_policy', loss_policy.item(), game_index)
        writer.add_scalar('entroy', entroy.item(), game_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

               
        [updated_action_policy, updated_value] = model(states)
        kl = F.kl_div(action_policy, updated_action_policy)
       
         
        if kl.item() > kl_threshold * 4:
            break 


""" def evaluate_plicy(model,encoder,basic_mcts_round_per_moves,az_mcts_round_per_moves,basic_mcts_temperature,the_device):
    msct_agent = MSCTAgent(1,"MSCTAgent","O",basic_mcts_round_per_moves,basic_mcts_temperature)
    az_agent = AlphaZeroAgent(2, "AZAent", "X", encoder, model, az_mcts_round_per_moves,device=the_device)

    number_of_games = 100    

    players = [msct_agent, az_agent]
    

    win_counts = {
        msct_agent.id: 0,
        az_agent.id: 0,
    }

    for game_index in tqdm(range(number_of_games)):
          
        agent_old.reset_memory()
        agent_new.reset_memory()
        
        winner=Connect5Game.run_episode(board_size,players,players[0 if game_index%2== 0 else 1],is_self_play=False)
        
        if winner is not None:
            win_counts[winner.id] += 1 """


        
def main():
    cfg = Utils.config()
    
    number_of_planes = cfg['GAME'].getint('number_of_planes')
    board_size = cfg['GAME'].getint('board_size')

    number_of_games = cfg['TRAIN'].getint('number_of_games')
    buffer_size = cfg['TRAIN'].getint('buffer_size')
    learning_rate = cfg['TRAIN'].getfloat('learning_rate')
    batch_size = cfg['TRAIN'].getint('batch_size')
    momentum_ = cfg['TRAIN'].getfloat('momentum')
    epochs = cfg['TRAIN'].getint('epochs')
    kl_threshold = cfg['TRAIN'].getfloat('kl_threshold')
    check_frequence= cfg['TRAIN'].getint('check_frequence')

    az_mcts_round_per_moves = cfg['AZ_MCTS'].getint('round_per_moves')

    basic_mcts_temperature= cfg['BASIC_MCTS'].getfloat('round_per_moves')

    encoder = MultiplePlaneEncoder(number_of_planes, board_size)

    writer = SummaryWriter()

    use_cuda = torch.cuda.is_available()
    the_device = torch.device('cuda' if use_cuda else 'cpu')
    
    input_shape = (number_of_planes, board_size, board_size)
    model = Connect5Network(input_shape, board_size * board_size)
    
    #if torch.cuda.device_count() > 1:
        #model = nn.DataParallel(model)
    
    model.to(the_device)    

    experience_collector_1 = AlphaZeroExperienceCollector()
    experience_collector_2 = AlphaZeroExperienceCollector()

    agent_1 = AlphaZeroAgent(1, "Agent1", "O", encoder, model, az_mcts_round_per_moves, experience_collector_1, device=the_device)
    agent_2 = AlphaZeroAgent(2, "Agent2", "X", encoder, model, az_mcts_round_per_moves, experience_collector_2, device=the_device)
    players = [agent_1, agent_2]

    experience_buffer = AlphaZeroExpericenceBuffer(buffer_size)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_)
    
    best_win_ratio = 0
    basic_mcts_round_per_moves= az_mcts_round_per_moves


    for game_index in tqdm(range(1, number_of_games)):

        # collect data via self-playing
        collect_data(agent_1, agent_2, board_size, players, game_index, experience_buffer)

        if experience_buffer.size() > batch_size:
           # update the policy with SGD
           improve_policy(experience_buffer, game_index, model, optimizer, batch_size, epochs, kl_threshold, the_device, writer)
           

        """ if game_index % check_frequence == 0:
            print("current self-play batch: {}".format(game_index+1))
            win_ratio = evaluate_plicy(model, basic_mcts_round_per_moves,az_mcts_round_per_moves,basic_mcts_temperature)
            torch.save(model.state_dict(), './current_policy.model')
                      
            if win_ratio > best_win_ratio:
                print("New best policy!!!!!!!!")
                best_win_ratio = win_ratio
                # update the best_policy
                torch.save(model.state_dict(),'./best_policy.model')
                if (best_win_ratio == 1.0 and basic_mcts_round_per_moves < 5000):
                    basic_mcts_round_per_moves *= 2
                    best_win_ratio = 0.0  """


if __name__ == '__main__':
    main()
