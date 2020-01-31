# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /

import argparse
import logging
import os
import sys

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent.experiencebuffer import ExpericenceBuffer
from boardencoder.blackwhiteencoder import BlackWhiteEncoder
from boardencoder.deepmindencoder import DeepMindEncoder
from boardencoder.snapshotencoder import SnapshotEncoder
from common.utils import Utils
from datacollector import DataCollector
from models.ResNet8Network import ResNet8Network
from models.Simple5Network import Simple5Network
from policychecker import PolicyChecker
from policyimprover import PolicyImprover


class Trainer:
    def __init__(self, args):
        
        self._logger = logging.getLogger('Traniner')
        self._checkpoint = None
        
        config_name = args.config

        # hardcode the config path just for convinence.
        cfg = Utils.config("./config/" + config_name)

        number_of_planes = cfg['GAME'].getint('number_of_planes')
        board_size = cfg['GAME'].getint('board_size')
        encoder_name = cfg['GAME'].get('encoder_name')

        az_mcts_rounds_per_move = cfg['AZ_MCTS'].getint('rounds_per_move')
        az_mcts_temperature = cfg['AZ_MCTS'].getfloat('temperature')
        c_puct = cfg['AZ_MCTS'].getfloat('C_puct')

        basic_mcts_c_puct = cfg['BASIC_MCTS'].getfloat('C_puct')
        
      

        buffer_size = cfg['TRAIN'].getint('buffer_size')
        
        batch_size = cfg['TRAIN'].getint('batch_size')

        epochs = cfg['TRAIN'].getint('epochs')
        kl_threshold = cfg['TRAIN'].getfloat('kl_threshold')
        
        self._basic_mcts_rounds_per_move = cfg['BASIC_MCTS'].getint('rounds_per_move')
        self._latest_checkpoint_file = './checkpoints/' + config_name.split('.')[0]+'/latest.pth.tar'
        self._best_checkpoint_file = './checkpoints/' +   config_name.split('.')[0]+'/best.pth.tar'

        check_number_of_games = cfg['EVALUATE'].getint('number_of_games')

        os.makedirs(os.path.dirname(self._latest_checkpoint_file), exist_ok=True)
        os.makedirs(os.path.dirname(self._best_checkpoint_file), exist_ok=True)

        use_cuda = torch.cuda.is_available()
        devices_ids = []
        if use_cuda:
            devices_ids = list(map(int, args.gpu_ids.split(',')))
            num_devices = torch.cuda.device_count()
            if len(devices_ids) > num_devices:
                raise Exception('#available gpu : {} < --device_ids : {}'.format(num_devices, len(devices_ids)))

        if encoder_name == 'SnapshotEncoder':
            encoder = SnapshotEncoder(number_of_planes, board_size)
            input_shape = (number_of_planes, board_size, board_size)

        if encoder_name == 'DeepMindEncoder':
            encoder = DeepMindEncoder(number_of_planes, board_size)
            input_shape = (number_of_planes*2+1, board_size, board_size)

        if encoder_name == 'BlackWhiteEncoder':
            encoder = BlackWhiteEncoder(number_of_planes, board_size)
            input_shape = (number_of_planes*2+2, board_size, board_size)

        self._model_name = cfg['MODELS'].get('net')
        self._model = ResNet8Network(input_shape, board_size * board_size) if self._model_name == 'ResNet8Network' else Simple5Network(
            input_shape, board_size * board_size)

        self._optimizer = Utils.get_optimizer(self._model.parameters(), cfg)
        
        
        self._experience_buffer = ExpericenceBuffer(buffer_size)
        self._check_frequence = cfg['TRAIN'].getint('check_frequence')
        
        self._start_game_index = 1
        self._train_number_of_games = cfg['TRAIN'].getint('number_of_games')

       
        # Be aware this is not the first time to run this program
        resume = args.resume
        if resume:
            self._checkpoint = torch.load(self._latest_checkpoint_file, map_location='cpu')
            if self._checkpoint['model_name'] == self._model_name:
                if use_cuda:
                    self._model.to(torch.device('cuda:'+str(devices_ids[0])))
                else:
                    self._model.to(torch.device('cpu'))

                self._model.load_state_dict(self._checkpoint['model'])
                self._optimizer.load_state_dict(self._checkpoint['optimizer'])
                self._basic_mcts_rounds_per_move = self._checkpoint['basic_mcts_rounds_per_move']
                
                self._start_game_index = self._checkpoint['game_index']
                self._experience_buffer.data = self._checkpoint['experience_buffer'].data
                self._logger.debug('ExpericenceBuffer size is {} when loading from checkpoint'.format(self._experience_buffer.size()))
        
        writer = SummaryWriter(log_dir='./runs/' + config_name.split('.')[0])

        self._data_collector = DataCollector(encoder, self._model, az_mcts_rounds_per_move, c_puct, az_mcts_temperature,
                                       board_size, number_of_planes, devices_ids, use_cuda)

        self._policy_improver = PolicyImprover(self._model, batch_size, epochs,kl_threshold,devices_ids, use_cuda, self._optimizer, writer)

        self._policy_checker = PolicyChecker(devices_ids, use_cuda, encoder, board_size, number_of_planes, self._model, az_mcts_rounds_per_move,
                                           c_puct, az_mcts_temperature, basic_mcts_c_puct, check_number_of_games, writer)



    def run(self):

        mp.set_start_method('spawn', force=True)
        self._checkpoint['model_name'] = self._model_name
        
        best_ratio = 0.0  
        for game_index in tqdm(range(self._start_game_index, self._train_number_of_games+1), desc='Training Loop'):
            self._checkpoint['game_index'] = game_index
            
            # collect data via self-playing
            collected_data = self._data_collector.collect_data(game_index)
            self._experience_buffer.merge(collected_data)
            self._checkpoint['experience_buffer'] = self._experience_buffer

            # update the policy
            self._policy_improver.improve_policy(game_index, self._experience_buffer)
            self._checkpoint['model'] = self._model.state_dict()
            self._checkpoint['optimizer']= self._optimizer.state_dict()

            # check the policy 
            if game_index % self._check_frequency == 0:
                win_ratio = self._policy_checker.check_policy(game_index, self._basic_mcts_rounds_per_move)
                self._checkpoint['basic_mcts_rounds_per_move'] = self._basic_mcts_rounds_per_move
                self._checkpoint['best_score'] = win_ratio
                
                # save the latest policy
                torch.save(self._checkpoint, self._latest_checkpoint_file)
                if win_ratio > best_ratio:
                    best_ratio = win_ratio
                    self._logger.info("New best score {:.2%} against MCTS {} rounds per move ".format(win_ratio, self._basic_mcts_rounds_per_move))

                    # save the best_policy
                    torch.save(self._checkpoint, self._best_checkpoint_file)
                    if (best_ratio > 0.8 and self._basic_mcts_rounds_per_move < 10000):
                        self._basic_mcts_rounds_per_move += 1000
                        best_ratio = 0.0


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Training')

    parser.add_argument('--gpu_ids', type=str, default='0',
                        help="Specifiy which gpu devices to use if available,e.g. '0,1,2'")

    parser.add_argument('--resume', type=bool, default=False,
                        help='Wethere resume traning from the previous or not ')

    parser.add_argument('-config', type=str, default='default.ini',
                        help='A ini config file to setup the default machinery')

    parser.add_argument('--debug', type=str, default='True',
                        help='which level information should be shown,INFO,DEBUG,')

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout,level=logging.DEBUG if args.debug else logging.INFO)

    Trainer(args).run()


if __name__ == '__main__':
    main()
