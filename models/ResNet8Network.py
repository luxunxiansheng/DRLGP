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


import numpy as np
import torch
import torch.nn as nn

NUM_FILTERS = 64


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1):
        super().__init__()

        self._conv = nn.Sequential(ConvBlock.layer_init(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding)),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self._conv(x)

    @staticmethod
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer


class ResNet8Network(nn.Module):
    def __init__(self, input_shape, num_points):
        super().__init__()

        self._conv_in = ConvBlock(input_shape[0], NUM_FILTERS)
        self._conv_1 = ConvBlock(NUM_FILTERS, NUM_FILTERS)
        self._conv_2 = ConvBlock(NUM_FILTERS, NUM_FILTERS)
        self._conv_3 = ConvBlock(NUM_FILTERS, NUM_FILTERS)
        self._conv_4 = ConvBlock(NUM_FILTERS, NUM_FILTERS)
        self._conv_5 = ConvBlock(NUM_FILTERS, NUM_FILTERS)

        body_out_shape = (NUM_FILTERS, ) + input_shape[1:]

        self._conv_value = ConvBlock(NUM_FILTERS, 1, kernel_size=1)
        conv_value_size = self._get_conv_value_size(body_out_shape)
        self._value_out = nn.Sequential(
            nn.Linear(conv_value_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

        self._conv_policy = ConvBlock(NUM_FILTERS, 2, kernel_size=1)
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self._policy_out = nn.Sequential(
            nn.Linear(conv_policy_size, num_points),
            nn.Softmax(dim=1)
        )

    def forward(self, encoded_boards):
        batch_size = encoded_boards.size()[0]
        v = self._conv_in(encoded_boards)
        v = v + self._conv_1(v)
        v = v + self._conv_2(v)
        v = v + self._conv_3(v)
        v = v + self._conv_4(v)
        processed_board = v + self._conv_5(v)

        # value head
        value = self._conv_value(processed_board)
        value = self._value_out(value.view(batch_size, -1))

        # policy head
        policy = self._conv_policy(processed_board)
        policy = self._policy_out(policy.view(batch_size, -1))

        return policy, value

    def _get_conv_value_size(self, shape):
        output = self._conv_value(torch.zeros(1, *shape))
        return int(np.prod(output.size()))

    def _get_conv_policy_size(self, shape):
        output = self._conv_policy(torch.zeros(1, *shape))
        return int(np.prod(output.size()))
