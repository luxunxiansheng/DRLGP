import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.ResNet8Network import ResNet8Network
from models.Simple5Network import Simple5Network
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent.alphazeroagent import (AlphaZeroAgent, AlphaZeroExpericenceBuffer,
                                  AlphaZeroExperienceCollector,
                                  MultiplePlaneEncoder)
from agent.mctsagent import MCTSAgent
from common.board import Board
from common.utils import Utils
from game.connect5game import Connect5Game


class Trainer(object):
    def __init__(self, args,logger):

        self._logger = logger   
        
        config_name = args.config
        
        # hardcode the config path just for convinence. 
        cfg = Utils.config("./config/" + config_name)
        
        
        self._number_of_planes = cfg['GAME'].getint('number_of_planes')
        self._board_size = cfg['GAME'].getint('board_size')

        self._az_mcts_round_per_moves = cfg['AZ_MCTS'].getint('round_per_moves')
        self._az_mcts_temperature = cfg['AZ_MCTS'].getfloat('temperature')

        self._basic_mcts_temperature = cfg['BASIC_MCTS'].getfloat('temperature')
        self._basic_mcts_round_per_moves = cfg['BASIC_MCTS'].getint('round_per_moves')

        self._train_number_of_games = cfg['TRAIN'].getint('number_of_games')
        self._buffer_size = cfg['TRAIN'].getint('buffer_size')
        self._learning_rate = cfg['TRAIN'].getfloat('learning_rate')
        self._batch_size = cfg['TRAIN'].getint('batch_size')

        self._epochs = cfg['TRAIN'].getint('epochs')
        self._kl_threshold = cfg['TRAIN'].getfloat('kl_threshold')
        self._check_frequence = cfg['TRAIN'].getint('check_frequence')

        
        self._current_model_file = './checkpoints/'+config_name.split('.')[0]+'/current.model'
        self._best_model_file    = './checkpoints/'+config_name.split('.')[0]+'/best.model'

        self._evaluate_number_of_games = cfg['EVALUATE'].getint('number_of_games')

        os.makedirs(os.path.dirname(self._current_model_file), exist_ok=True)
        os.makedirs(os.path.dirname(self._best_model_file), exist_ok=True)

        use_cuda = torch.cuda.is_available()

        gpu_ids = list(map(int, args.gpu_ids.split(',')))
        num_devices = torch.cuda.device_count()
        if num_devices < len(gpu_ids):
            raise Exception('#available gpu : {} < --device_ids : {}'.format(num_devices, len(gpu_ids)))
        cuda = 'cuda:' + str(gpu_ids[0])
        self._device = torch.device(cuda if use_cuda else 'cpu')

        experience_collector_1 = AlphaZeroExperienceCollector()
        experience_collector_2 = AlphaZeroExperienceCollector()
        self._encoder = MultiplePlaneEncoder(self._number_of_planes, self._board_size)

        input_shape = (self._number_of_planes, self._board_size, self._board_size)
        model_name = cfg['MODELS'].get('net')
        self._model = ResNet8Network(input_shape, self._board_size * self._board_size) if model_name == 'ResNet8Network' else Simple5Network(input_shape, self._board_size * self._board_size)

        # Be aware this is not the first time to run this program
        resume = args.resume
        if resume:
            self._model.load_state_dict(torch.load(self._current_model_file))

        #self._model = DataParallel(self._model,device_ids=gpu_ids)
        self._model = self._model.to(self._device)

        self._agent_1 = AlphaZeroAgent(0, "Agent1", "O", self._encoder, self._model, self._az_mcts_round_per_moves, self._az_mcts_temperature, experience_collector_1, device=self._device)
        self._agent_2 = AlphaZeroAgent(1, "Agent2", "X", self._encoder, self._model, self._az_mcts_round_per_moves, self._az_mcts_temperature, experience_collector_2, device=self._device)

        self._experience_buffer = AlphaZeroExpericenceBuffer(self._buffer_size)

        self._optimizer = Utils.get_optimizer(self._model.parameters(), cfg)

        self._writer = SummaryWriter(log_dir='./runs/'+config_name.split('.')[0])

    def _collect_data(self, game_index):
        self._agent_1.reset()
        self._agent_2.reset()

        players = [self._agent_1, self._agent_2]

        winner = Connect5Game.run_episode(self._board_size, players, players[0 if game_index % 2 == 0 else 1],  self._number_of_planes, True)

        if winner is not None:
            if winner == players[0]:
                players[0].experience_collector.complete_episode(reward=1)
                players[1].experience_collector.complete_episode(reward=-1)
            if winner == players[1]:
                players[1].experience_collector.complete_episode(reward=1)
                players[0].experience_collector.complete_episode(reward=-1)

            self._experience_buffer.combine_experience([self._agent_1.experience_collector, self._agent_2.experience_collector])

    def _improve_policy(self, game_index):
        self._model.train()

        batch_data = random.sample(self._experience_buffer.data, self._batch_size)

        for _ in range(self._epochs):
            states, rewards, visit_counts = zip(*batch_data)

            states = torch.from_numpy(np.array(list(states))).to(self._device, dtype=torch.float)
            rewards = torch.from_numpy(np.array(list(rewards))).to(self._device, dtype=torch.float)
            visit_counts = torch.from_numpy(np.array(list(visit_counts))).to(self._device, dtype=torch.float)

            action_policy_target = F.softmax(visit_counts, dim=1)
            value_target = rewards

            [action_policy, value] = self._model(states)

            log_action_policy = torch.log(action_policy)
            loss_policy = - log_action_policy * action_policy_target
            loss_policy = loss_policy.sum(dim=1).mean()

            loss_value = F.mse_loss(value.squeeze(), value_target)

            loss = loss_policy + loss_value

            with torch.no_grad():
                entroy = -torch.mean(torch.sum(log_action_policy*action_policy, dim=1))

            self._writer.add_scalar('loss', loss.item(), game_index)
            self._writer.add_scalar('loss_value', loss_value.item(), game_index)
            self._writer.add_scalar('loss_policy', loss_policy.item(), game_index)
            self._writer.add_scalar('entropy', entroy.item(), game_index)

            self._optimizer.zero_grad()

            loss.backward()
            self._optimizer.step()
            [updated_action_policy, _] = self._model(states)
            kl = F.kl_div(action_policy, updated_action_policy).item()

            if kl > self._kl_threshold * 4:
                break

    def _evaluate_plicy(self):
        mcts_agent = MCTSAgent(0, "MCTSAgent", "O", self._basic_mcts_round_per_moves, self._basic_mcts_temperature)
        az_agent = AlphaZeroAgent(1, "AZAent", "X", self._encoder, self._model, self._az_mcts_round_per_moves, self._az_mcts_temperature, device=self._device)

        win_counts = {
            mcts_agent.id: 0,
            az_agent.id: 0,
        }

        for game_index in tqdm(range(self._evaluate_number_of_games)):
            az_agent.reset()
            players = [mcts_agent, az_agent]
            winner = Connect5Game.run_episode(self._board_size, players, players[0 if game_index % 2 == 0 else 1], self._number_of_planes, is_self_play=False)

            if winner is not None:
                win_counts[winner.id] += 1

        self._logger.info('mcts:az_agent---{}:{} in {}'.format(win_counts[mcts_agent.id], win_counts[az_agent.id], self._evaluate_number_of_games))

        return win_counts[az_agent.id]/self._evaluate_number_of_games

    def run(self):

        best_win_ratio = 0

        for game_index in tqdm(range(1, self._train_number_of_games)):
            # collect data via self-playing
            self._collect_data(game_index)

            if self._experience_buffer.size() > self._batch_size:
                # update the policy with SGD
                self._improve_policy(game_index)

            if game_index % self._check_frequence == 0:
                
                win_ratio = self._evaluate_plicy()

                self._logger.info("current self-play batch:{} and win win ratio is:{}".format(game_index, win_ratio))

                torch.save(self._model.state_dict(), self._current_model_file)

                if win_ratio > best_win_ratio:
                    self._logger.info("New best policy!!!!!!!!")
                    best_win_ratio = win_ratio
                    # update the best_policy
                    torch.save(self._model.state_dict(), self._best_model_file)
                    if (best_win_ratio == 1.0 and self._basic_mcts_round_per_moves < 6000):
                        self._basic_mcts_round_per_moves += 1000
                        best_win_ratio = 0.0


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Training')
    parser.add_argument('--gpu_ids', type=str, default='0', help="Specifiy which gpu devices to use if available,e.g. '0,1,2'")
    parser.add_argument('--resume', type=bool, default=False, help='Wethere resume traning from the previous or not ')
    parser.add_argument('--config', type = str,default='default.ini',help='A ini config file to setup the default machinery')
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)

    Trainer(args,logger).run()


if __name__ == '__main__':
    main()
