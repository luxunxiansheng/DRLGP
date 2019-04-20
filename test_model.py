import numpy as np

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._fc1 = nn.Linear(9, 20)
        self._fc2 = nn.Linear(20, 40)
        self._fc3 = nn.Linear(40, 9)

    def forward(self, x):
        x = torch.sigmoid(self._fc1(x))
        x = torch.sigmoid(self._fc2(x))
        x = torch.sigmoid(self._fc3(x))

        return x


def main():
    model= Net()
    model.load_state_dict(torch.load('./checkpoints/ttt3_mlp.pth.tar'))
    model.eval()

    test_board=np.array([[
           -1,-1,  0,
            1, 0,  0,  
            0, 0,  0,

    ]])

    move_prob=model(torch.tensor(test_board,dtype=torch.float))[0]

    i = 0

    for row in range(3):
        row_format =[]
        for col in range(3):
            row_format.append('{:.3f} '.format(move_prob[i]))
            i +=1
        print(''.join(row_format))    


if __name__ == "__main__":
    main()