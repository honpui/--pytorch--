import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

data1=torch.ones(100,2)
x0=torch.normal(2*data1,1)
# print(x0.size()[0])
y0=torch.ones(x0.size()[0])
# print(y0.size())
x1=torch.normal(-2*data1,1)
y1=torch.zeros(x1.size()[0])

x=torch.cat((x0,x1),0)
y=torch.cat((y0,y1),0).type(torch.LongTensor)#crossentropy要求是longtensor 类型
# print(y.size())

# print(x0.size())

# plt.scatter(x0.numpy()[:,0],x0.numpy()[:,1],c=y0.numpy())
# plt.scatter(x1.numpy()[:,0],x1.numpy()[:,1],c=y1.numpy())
# plt.scatter(x.numpy()[:,0],x.numpy()[:,1],c=y.numpy())
# plt.show()
x=Variable(x)
y=Variable(y)

net=nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,10),
    nn.ReLU(),
    nn.Linear(10,2)
)

optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
lossfunc=nn.CrossEntropyLoss()


plt.ion()

for i in range(100):
    pred = net(x)
    # print(pred.size())
    predition=torch.max(pred,1)[1]
    # print(predition.size())

    target_y=predition.data.numpy().squeeze()
    # print(pred.size())
    # print(y.size())
    optimizer.zero_grad()
    loss=lossfunc(pred,y)
    loss.backward()
    optimizer.step()

    if i%1==0:
        print('step:',i,'|loss:',loss.data[0])
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=target_y)#此处c的取值要求是1或者0，这样能够将不同的类分开，要求是numpy类型
        plt.pause(0.1)


plt.ioff()
plt.show()