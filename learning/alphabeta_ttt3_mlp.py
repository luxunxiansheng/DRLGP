import numpy as np
import torch
import torch.nn as nn


 class Net(nn.Module):
     def __init__(self):
         self._fc = nn.Linear()
         




 boards = np.load('../generated_data/features.npy')
 moves = np.load('../generated_data/labels.npy')
 
 samples = boards.shape[0]

 X = boards.reshape(samples, 9)
 Y = moves.reshape(samples, 9)
 
 train_samples=int(0.9*samples)

X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]





