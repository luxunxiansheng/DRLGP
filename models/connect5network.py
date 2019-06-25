import numpy as np
import torch
import torch.nn as nn

NUM_FILTERS = 64


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer



class Connect5Network(nn.Module):
    def __init__(self,input_shape, num_points):
        super().__init__()

        self._conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )

        
        self._conv_1 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self._conv_2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self._conv_3 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self._conv_4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self._conv_5 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )

        body_out_shape = (NUM_FILTERS, ) + input_shape[1:]

        # value head
        self._conv_value = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        conv_value_size = self._get_conv_val_size(body_out_shape)
        self._value_out = nn.Sequential(
            nn.Linear(conv_value_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

        # policy head
        self._conv_policy = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU()
        )
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self._policy_out = nn.Sequential(
            nn.Linear(conv_policy_size, num_points)
        )

    def _get_conv_value_size(self, shape):
        output = self.conv_value(torch.zeros(1, *shape))
        return int(np.prod(output.size()))

    def _get_conv_policy_size(self, shape):
        output = self.conv_policy(torch.zeros(1, *shape))
        return int(np.prod(output.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        v = self._conv_in(x)
        v = v + self._conv_1(v)
        v = v + self._conv_2(v)
        v = v + self._conv_3(v)
        v = v + self._conv_4(v)
        v = v + self._conv_5(v)
        value = self._conv_value(v)
        value = self._value_out(value.view(batch_size, -1))
        
        policy = self._conv_policy(v)
        policy = self._policy_out(policy.view(batch_size, -1))
        return policy, value














































class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        self._conv = nn.Sequential(layer_init(nn.Conv2d(input_channels, output_channels, kernel_size, stride)), nn.ReLU())

    def forward(self, x):
        return self._conv(x)

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0],-1)
        return x

class DynamicDense(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self._out_features = out_features
     
 

    def forward(self, x):
        if not hasattr(self, '_linear'):
            input_dim = x.size()[1]
            device= x.device
            self._linear = nn.Linear(input_dim, self._out_features).to(device)
    
        return self._linear(x)
        


class Connect5Network(nn.Module):
    def __init__(self, input_channels, num_points):
        super().__init__()

        self._conv_layers = nn.Sequential(
            ConvBlock(input_channels, 64, 3, 1),
            ConvBlock(64, 64, 3, 1),
            ConvBlock(64, 64, 3, 1),
            Flatten(),
            DynamicDense(512)
        )

        self._policy_hidden_layer = nn.Sequential(DynamicDense(512), nn.ReLU())
        self._policy_output_layer = nn.Sequential(DynamicDense(num_points), nn.Softmax())

        self._value_hidden_layer = nn.Sequential(DynamicDense(512), nn.ReLU())
        self._value_output_layer = nn.Sequential(DynamicDense(1), nn.Tanh())

    def forward(self, encoded_boards):
        processed_board = self._conv_layers(encoded_boards)

        output_policy_hidden = self._policy_hidden_layer(processed_board)
        output_policy = self._policy_output_layer(output_policy_hidden)

        output_value_hidden = self._value_hidden_layer(processed_board)
        output_value = self._value_output_layer(output_value_hidden)

        return output_policy, output_value
