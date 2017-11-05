import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-1,1,100),1)
y=x.pow(2)+0.2*torch.randn(x.size())

x,y=Variable(x),Variable(y)

net=nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,1)
)


optimizer=torch.optim.Adam(net.parameters(),lr=0.05)
loss_func=torch.nn.MSELoss()
plt.ion()

for i in range(500):
    pred=net(x)
    loss = loss_func(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%5==0:
        plt.cla()
        print('step:',i,'|loss:',loss.data[0])
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),pred.data.numpy())
        plt.pause(0.1)

plt.ioff()
plt.show()


