import torch
import torch.utils.data as Dataset
import numpy as np
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

epoch=1
batch_size=50
lr=0.01

traindata=torchvision.datasets.MNIST(
    root='./mnist',
    download=False,
    transform=torchvision.transforms.ToTensor(),
    train=True

)

testdata=torchvision.datasets.MNIST(root='./mnist',train=False)

trainload=Dataset.DataLoader(traindata,batch_size=batch_size,shuffle=True)

test_x=Variable(torch.unsqueeze(testdata.test_data[:200],1).type(torch.FloatTensor))
test_y=testdata.test_labels[:200].type(torch.FloatTensor)

# print(test_x.size())

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size()[0],32*7*7)#此处可能有问题
        x=self.out(x)
        return x

cnn=CNN()
optimizer=torch.optim.Adam(cnn.parameters(),lr=lr)
lossfunc=torch.nn.CrossEntropyLoss()


plt.ion()
for step,(x,y) in enumerate(trainload):
    x=Variable(x)
    y=Variable(y)
    pred=cnn(x)
    loss=lossfunc(pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step%100==0:
        test_pred=cnn(test_x)
        test_pred=torch.max(test_pred,1)[1].type(torch.FloatTensor)

        # print(test_pred)
        accuracy=torch.mean((test_pred.data==test_y).type(torch.FloatTensor))
        print('loss:',loss.data[0],'|accuracy:',accuracy)
