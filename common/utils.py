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


import errno
import os
import shutil
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop


class Utils(object):
    @staticmethod
    def gpu_id_with_max_memory():
        os.system('nvidia-smi -q -d Memory|grep -A4 GPU|grep Free > dump')
        memory_available = [int(x.split()[2]) for x in open('dump', 'r').readlines()]
        os.system('rm ./dump')
        return np.argmax(memory_available)

    @staticmethod
    def config(cfg_path):
        # parser config
        config = ConfigParser()
        config.read(cfg_path)
        return config

    @staticmethod
    def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
        """
        Learning rate schedule with respect to epoch
        lr: float, initial learning rate
        lr_factor: float, decreasing factor every epoch_lr
        epoch_now: int, the current epoch
        lr_epochs: list of int, decreasing every epoch in lr_epochs
        return: lr, float, scheduled learning rate.
        """
        count = 0
        for epoch in lr_epochs:
            if epoch_now >= epoch:
                count += 1
                continue
            break

        return lr * np.power(lr_factor, count)

    @staticmethod
    def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(num_features, eps=eps)
        elif norm_type == "GroupNorm":
            return nn.GroupNorm(num_groups, num_features, eps=eps)
        elif norm_type == "InstanceNorm":
            return nn.InstanceNorm2d(num_features, eps=eps, affine=True, track_running_stats=True)
        else:
            raise Exception('Unknown Norm Function : {}'.format(norm_type))

    @staticmethod
    def get_optimizer(params, cfg):
        if cfg['TRAIN'].get('optimizer') == 'Adam':
            return Adam(params, lr=cfg['TRAIN.OPTIMIZER.ADAM'].getfloat('learning_rate'), weight_decay=cfg['TRAIN.OPTIMIZER.ADAM'].getfloat('weight_decay'))
        else:
            raise Exception('Unknown optimizer : {}'.format(cfg['TRAIN'].get('optimizer')))
