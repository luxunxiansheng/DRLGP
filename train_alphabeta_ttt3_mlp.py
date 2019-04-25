import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models.feedfowrdnerualnetwork import FeedForwardNeuralNetwork


def train_batch(epach,model,optimizer,mini_batch,device='cpu'):    
    train_loss = 0
    train_correct = 0

    optimizer.zero_grad()
 
    X= torch.tensor(np.array(mini_batch)[:,0,:],dtype=torch.float).to(device)
    Y= torch.tensor(np.array(mini_batch)[:,1,:],dtype=torch.float).to(device) 
   
    output = model(X)
    train_loss = F.mse_loss(output,Y,reduction='sum')
    
    pred = output.argmax(dim=1, keepdim=True)
    target = Y.argmax(dim=1, keepdim=True)
    
    train_correct += pred.eq(target.view_as(pred)).sum().item()
        
    train_loss.backward()
    optimizer.step()
   
    return train_correct
    
def test(epoch,model,test_data,device='cpu'):
    test_loss = 0
    test_correct = 0

    X= torch.tensor(np.array(test_data)[:,0,:],dtype=torch.float).to(device)
    Y= torch.tensor(np.array(test_data)[:,1,:],dtype=torch.float).to(device)

    with torch.no_grad():
        output = model(X)
        test_loss += F.mse_loss(output,Y,reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        target = Y.argmax(dim=1, keepdim=True)
        
        test_correct += pred.eq(target.view_as(pred)).sum().item()
   

    print("Epach:{},Loss:{}".format(epoch,test_loss))

    return test_correct

def main():
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    batch_size = 32

    model = FeedForwardNeuralNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5,weight_decay=0.01)

    boards = np.load('./generated_data/features.npy')
    moves =  np.load('./generated_data/labels.npy')

    sample_num = boards.shape[0]

    X = boards.reshape(sample_num, 9)
    Y = moves.reshape(sample_num, 9)

    sample_data = [[board, move] for board, move in zip(X, Y)]

    train_num = int(0.9*sample_num)

    train_data = sample_data[:train_num]
    test_data = sample_data[train_num:]

    for epoch in tqdm(range(1,200)):
        
        train_correct = 0
        
        random.shuffle(train_data)
        train_batches = [train_data[k:k+batch_size] for k in range(0, train_num, batch_size)]

        model.train()
        for _,mini_batch in tqdm(enumerate(train_batches)):
            train_correct+=train_batch(epoch,model,optimizer,mini_batch,device)
        print('Train Epoch: {}, Train Accuracy:{:.0f}%'.format(epoch,100.*train_correct/train_num))       

        model.eval()
        test_correct= test(epoch,model,test_data,device)
        print('Test Epoch: {}, test Accuracy:{:.0f}%'.format(epoch,100.*test_correct/len(test_data)))     

    torch.save(model.state_dict(),'./checkpoints/ttt3_mlp.pth.tar')      

if __name__ == '__main__':
    main()
