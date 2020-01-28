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

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel


class PolicyImprover:
    def __init__(self, model, model_name, batch_size, epochs, kl_threshold, expericence_buffer, devices_ids, use_cuda, optimizer, writer, checkpoint, logger):
        self._use_cuda = use_cuda
        self._devices_ids = devices_ids
        self._model = model
        self._model_name = model_name,
        self._batch_size = batch_size,
        self._epochs = epochs
        self._experience_buffer = expericence_buffer
        self._kl_threshold = kl_threshold
        self._logger = logger
        self._optimizer = optimizer
        self._entropy = 0
        self._loss = 0
        self._loss_value = 0
        self._loss_policy = 0
        self._writer = writer
        self._checkpoint = checkpoint

        if use_cuda:
            self._devices = [torch.device(
                'cuda:'+str(devices_ids[i])) for i in range(len(devices_ids))]
        else:
            self._devices = [torch.device('cpu')]

    def improve_policy(self, game_index):
        self._model.train()

        if self._use_cuda:
            device = self._devices[0]
            self._model = DataParallel(self._model.to(
                device), device_ids=self._devices_ids)
        else:
            self._model.to(device)

        batch_data = random.sample(
            self._experience_buffer.data, self._batch_size)
        for _ in range(self._epochs):
            states, rewards, visit_counts = zip(*batch_data)
            states = torch.from_numpy(np.array(list(states))).to(
                device, dtype=torch.float)
            rewards = torch.from_numpy(np.array(list(rewards))).to(
                device, dtype=torch.float)
            visit_counts = torch.from_numpy(
                np.array(list(visit_counts))).to(device, dtype=torch.float)

            action_policy_target = F.softmax(visit_counts, dim=1)
            value_target = rewards

            [action_policy, value] = self._model(states)

            log_action_policy = torch.log(action_policy)
            self._loss_policy = - log_action_policy * action_policy_target
            self._loss_policy = self._loss_policy.sum(dim=1).mean()

            self._loss_value = F.mse_loss(value.squeeze(), value_target)

            self._loss = self._loss_policy + self._loss_value

            with torch.no_grad():
                self._entropy = - \
                    torch.mean(
                        torch.sum(log_action_policy*action_policy, dim=1))

            self._optimizer.zero_grad()

            self._loss.backward()
            self._optimizer.step()
            [updated_action_policy, _] = self._model(states)
            kl = F.kl_div(action_policy, updated_action_policy).item()

            if kl > self._kl_threshold * 4:
                break

        real_game_index = game_index * \
            len(self._devices) if len(self._devices) > 1 else game_index

        self._writer.add_scalar('loss', self._loss.item(), real_game_index)
        self._writer.add_scalar(
            'loss_value', self._loss_value.item(), real_game_index)
        self._writer.add_scalar(
            'loss_policy', self._loss_policy.item(), real_game_index)
        self._writer.add_scalar(
            'entropy', self._entropy.item(), real_game_index)

        # refer to https://discuss.pytorch.org/t/how-could-i-train-on-multi-gpu-and-infer-with-single-gpu/22838/7

        if self._use_cuda:
            self._model = self._model.module.to(torch.device('cpu'))

        self._checkpoint = {'game_index': game_index,
                            'model_name': self._model_name,
                            'model': self._model.state_dict(),
                            'optimizer': self._optimizer.state_dict(),
                            'entropy': self._entropy.item(),
                            'loss': self._loss.item(),
                            'loss_policy': self._loss_policy.item(),
                            'loss_value': self._loss_value.item(),
                            'experience_buffer': self._experience_buffer
                            }

        self._logger.debug(
            '--Policy Improved  in round {}--'.format(game_index))
