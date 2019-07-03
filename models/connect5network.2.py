import torch
import torch.nn as nn


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


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

