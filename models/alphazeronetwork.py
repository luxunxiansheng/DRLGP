import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,stride):
        super().__init__()
        block = [nn.Conv2d(input_channels,output_channels,kernel_size,stride)]
        block += [nn.ReLU()]
        self._block = nn.Sequential(*block)
    
    def forward(self, x):
        return self._block(x)

class Flatten(nn.Module):
    def forward(self,x):
        x= x.view(x.size()[0],-1)
        return x 

class Dense(nn.Module):
    def __init__(self,out_features):
        super().__init__()
        self._out_features=out_features
        
    def forward(self,x):
        in_features= x.size()
        x= nn.Linear(in_features,self._out_features)
        return x        

class AlphaZeroNetwork(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        layers=[ConvBlock(input_channels,64,7,1)]
        layers+=[ConvBlock(64,64,5,1)]
        layers+=[ConvBlock(64,64,5,1)]
        layers+=[ConvBlock(64,48,5,1)]
        layers+=[ConvBlock(48,48,5,1)]
        layers+=[ConvBlock(48,32,5,1)]
        layers+=[ConvBlock(32,32,5,1)]
        
        layers+=[Flatten()]
        layers+=[Dense(1024)]
        layers+=[nn.ReLU()]

        self._net=nn.Sequential(*layers)

    def forward(self,x):
        return self._net(x)
