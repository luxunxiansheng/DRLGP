import numpy as np
import torch
import torch.nn as nn

NUM_FILTERS = 64


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1):
        super().__init__()

        self._conv = nn.Sequential(ConvBlock.layer_init(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding)),
            nn.ReLU()
        )

    def forward(self, x):
        return self._conv(x)

    @staticmethod
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer


class Simple5Network(nn.Module):
    def __init__(self, input_shape, num_points):
        super().__init__()

        self._conv_in = ConvBlock(input_shape[0], NUM_FILTERS)
        self._conv_1 = ConvBlock(NUM_FILTERS, NUM_FILTERS)
        self._conv_2 = ConvBlock(NUM_FILTERS, NUM_FILTERS)

        body_out_shape = (NUM_FILTERS,) + input_shape[1:]
        conv_2_out_size = self._get_con_2_out_size(body_out_shape)
        
        self._value_hidden_layer = nn.Sequential(nn.Linear(conv_2_out_size,512),nn.ReLU())
        self._value_output_layer = nn.Sequential(nn.Linear(512,1), nn.Tanh())
               
        self._policy_hidden_layer = nn.Sequential(nn.Linear(conv_2_out_size,512),nn.ReLU())
        self._policy_output_layer = nn.Sequential(nn.Linear(512,num_points), nn.Softmax(dim=1))
           
     

    def forward(self, encoded_boards):
        batch_size = encoded_boards.size()[0]
        v = self._conv_in(encoded_boards)
        v = self._conv_1(v)
        processed_board = self._conv_2(v)

        output_value_hidden = self._value_hidden_layer(processed_board.view(batch_size,-1))
        output_value = self._value_output_layer(output_value_hidden)

            

        output_policy_hidden = self._policy_hidden_layer(processed_board.view(batch_size,-1))
        output_policy = self._policy_output_layer(output_policy_hidden)

        return output_policy, output_value



    def _get_con_2_out_size(self, shape):
        output = self._conv_2(torch.zeros(1, *shape))
        return int(np.prod(output.size()))
 