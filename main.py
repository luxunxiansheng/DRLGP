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
from policyevaluator import PolicyEvaluator
from policyimprover import PolicyImprover


class Trainer:
    def __init__(self, args, logger):

        self._logger = logger

        config_name = args.config

        # hardcode the config path just for convinence.
        cfg = Utils.config("./config/" + config_name)

        self._number_of_planes = cfg['GAME'].getint('number_of_planes')
        self._board_size = cfg['GAME'].getint('board_size')
        self._encoder_name = cfg['GAME'].get('encoder_name')

        self._az_mcts_rounds_per_move = cfg['AZ_MCTS'].getint(
            'rounds_per_move')
        self._az_mcts_temperature = cfg['AZ_MCTS'].getfloat('temperature')
        self._c_puct = cfg['AZ_MCTS'].getfloat('C_puct')

        self._basic_mcts_c_puct = cfg['BASIC_MCTS'].getfloat(
            'C_puct')
        self._basic_mcts_rounds_per_move = cfg['BASIC_MCTS'].getint(
            'rounds_per_move')

        self._train_number_of_games = cfg['TRAIN'].getint('number_of_games')
        self._batch_of_self_play = cfg['TRAIN'].getint('batch_of_self_play')

        self._buffer_size = cfg['TRAIN'].getint('buffer_size')
        self._learning_rate = cfg['TRAIN'].getfloat('learning_rate')
        self._batch_size = cfg['TRAIN'].getint('batch_size')

        self._epochs = cfg['TRAIN'].getint('epochs')
        self._kl_threshold = cfg['TRAIN'].getfloat('kl_threshold')
        self._check_frequence = cfg['TRAIN'].getint('check_frequence')

        self._latest_checkpoint_file = './checkpoints/' + \
            config_name.split('.')[0]+'/latest.pth.tar'
        self._best_checkpoint_file = './checkpoints/' + \
            config_name.split('.')[0]+'/best.pth.tar'

        self._evaluate_number_of_games = cfg['EVALUATE'].getint(
            'number_of_games')

        os.makedirs(os.path.dirname(
            self._latest_checkpoint_file), exist_ok=True)
        os.makedirs(os.path.dirname(self._best_checkpoint_file), exist_ok=True)

        self._use_cuda = torch.cuda.is_available()
        self._devices_ids = []
        if self._use_cuda:
            self._devices_ids = list(map(int, args.gpu_ids.split(',')))
            num_devices = torch.cuda.device_count()
            if len(self._devices_ids) > num_devices:
                raise Exception(
                    '#available gpu : {} < --device_ids : {}'.format(num_devices, len(self._devices_ids)))

        if self._encoder_name == 'SnapshotEncoder':
            self._encoder = SnapshotEncoder(
                self._number_of_planes, self._board_size)
            input_shape = (self._number_of_planes,
                           self._board_size, self._board_size)

        if self._encoder_name == 'DeepMindEncoder':
            self._encoder = DeepMindEncoder(
                self._number_of_planes, self._board_size)
            input_shape = (self._number_of_planes*2+1,
                           self._board_size, self._board_size)

        if self._encoder_name == 'BlackWhiteEncoder':
            self._encoder = BlackWhiteEncoder(
                self._number_of_planes, self._board_size)
            input_shape = (self._number_of_planes*2+2,
                           self._board_size, self._board_size)

        self._model_name = cfg['MODELS'].get('net')
        self._model = ResNet8Network(input_shape, self._board_size * self._board_size) if self._model_name == 'ResNet8Network' else Simple5Network(
            input_shape, self._board_size * self._board_size)

        self._experience_buffer = ExpericenceBuffer(self._buffer_size)

        self._optimizer = Utils.get_optimizer(self._model.parameters(), cfg)

        self._start_game_index = 1

        self._entropy = 0
        self._loss = 0
        self._loss_value = 0
        self._loss_policy = 0

        # Be aware this is not the first time to run this program
        resume = args.resume
        if resume:
            self._checkpoint = torch.load(
                self._latest_checkpoint_file, map_location='cpu')
            if self._checkpoint['model_name'] == self._model_name:
                if self._use_cuda:
                    self._model.to(torch.device(
                        'cuda:'+str(self._devices_ids[0])))
                else:
                    self._model.to(torch.device('cpu'))

                self._model.load_state_dict(self._checkpoint['model'])
                self._optimizer.load_state_dict(self._checkpoint['optimizer'])
                self._start_game_index = self._checkpoint['game_index']
                self._entropy = self._checkpoint['entropy']
                self._loss = self._checkpoint['loss']
                self._loss_value = self._checkpoint['loss_value']
                self._loss_policy = self._checkpoint['loss_policy']
                self._experience_buffer.data = self._checkpoint['experience_buffer'].data
                self._basic_mcts_rounds_per_move = self._checkpoint['basic_mcts_rounds_per_move']
                self._logger.debug('ExpericenceBuffer size is {}'.format(
                    self._experience_buffer.size()))

        self._writer = SummaryWriter(
            log_dir='./runs/' + config_name.split('.')[0])

        self._checkpoint = None

    def run(self):

        mp.set_start_method('spawn', force=True)

        best_ratio = 0.0

        data_collector = DataCollector(self._encoder, self._model, self._az_mcts_rounds_per_move, self._c_puct, self._az_mcts_temperature,
                                       self._board_size, self._number_of_planes, self._experience_buffer, self._devices_ids, self._use_cuda, self._logger)

        policy_improver = PolicyImprover(self._model, self._model_name, self._batch_size, self._epochs, self._kl_threshold, self._experience_buffer,
                                         self._devices_ids, self._use_cuda, self._optimizer, self._writer, self._checkpoint, self._logger)

        policy_evaluator = PolicyEvaluator(self._devices_ids, self._use_cuda, self._encoder, self._board_size, self._number_of_planes, self._model,
                                           self._az_mcts_rounds_per_move, self._c_puct, self._az_mcts_temperature, self._basic_mcts_c_puct, self._basic_mcts_rounds_per_move, self._evaluate_number_of_games, self._logger)

        for game_index in tqdm(range(self._start_game_index, self._train_number_of_games+1), desc='Training Loop'):
            # collect data via self-playing

            data_collector.collect_data(game_index)

            if self._experience_buffer.size() > self._batch_size:
                # update the policy with SGD
                policy_improver.improve_policy(game_index)

                if game_index % self._check_frequence == 0:

                    win_ratio = policy_evaluator.evaluate_policy(game_index)

                    self._writer.add_scalar('score', win_ratio, game_index * len(
                        self._devices_ids) if len(self._devices_ids) > 1 else game_index)

                    self._checkpoint['basic_mcts_rounds_per_move'] = self._basic_mcts_rounds_per_move
                    self._checkpoint['best_score'] = win_ratio

                    torch.save(self._checkpoint, self._latest_checkpoint_file)

                    if win_ratio > best_ratio:
                        self._logger.info(
                            "New best score {:.2%}".format(win_ratio))

                        best_ratio = win_ratio

                        # update the best_policy
                        torch.save(self._checkpoint, self._best_checkpoint_file)
                        if (best_ratio == 1.0 and self._basic_mcts_rounds_per_move < 8000):
                            self._basic_mcts_rounds_per_move += 1000
                            self._logger.debug('current basic_mcts_round_moves:{}'.format(self._basic_mcts_rounds_per_move))
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

    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG if args.debug else logging.INFO)

    logger = logging.getLogger(__name__)

    Trainer(args, logger).run()


if __name__ == '__main__':
    main()
