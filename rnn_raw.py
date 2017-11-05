import numpy as np
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

lr=0.001


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.RNN(
            batch_first=True,
            input_size=1,
            hidden_size=32,
            num_layers=1

        )
        self.out=nn.Linear(32,1)

    def forward(self, x,hidden_s):
        x_out,hidden_s=self.rnn(x,hidden_s)
        out=[]
        for step in range(x_out.size(1)):
            out.append(self.out(x_out[:,step,:]))
        return torch.stack(out,dim=1),hidden_s

rnn=RNN()
optimizer=torch.optim.Adam(rnn.parameters(),lr=lr)
lossfunc=nn.MSELoss()


plt.ion()
hidden_s=None
for step in range(120):
    start,end=step*np.pi,(step+1)*np.pi
    # steps=np.linspace(start,end,10,dtype=np.float32)
    # print(type(steps))
    steps = np.linspace(start, end, 10)
    x_np=np.sin(steps)
    y_np=np.cos(steps)
    x=Variable(torch.from_numpy(x_np[np.newaxis,:,np.newaxis])).type(torch.FloatTensor)
    y=Variable(torch.from_numpy(y_np[np.newaxis,:,np.newaxis])).type(torch.FloatTensor)

    prediction,hidden_s=rnn(x,hidden_s)
    # print(prediction.size())

    hidden_s=Variable(hidden_s.data)

    loss=lossfunc(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')#继续调整此处的问题
    plt.pause(0.1)



plt.ioff()
plt.show()