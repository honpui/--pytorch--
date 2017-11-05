import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

traindata=torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor()

)
trainload=Data.DataLoader(traindata,batch_size=64,shuffle=True)


class AUTO(nn.Module):
    def __init__(self):
        super(AUTO,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3),
            nn.Tanh()

        )
        self.decoder=nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28)

        )
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

autoen=AUTO()
optimizer=torch.optim.Adam(autoen.parameters(),lr=0.01)
lossfunc=nn.MSELoss()



f,a=plt.subplots(2,5)
plt.ion()
viewdata=Variable(traindata.train_data[:5].view(-1,28*28)).type(torch.FloatTensor)
# print(viewdata.size())
for i in range(5):
    a[0][i].imshow(np.reshape(viewdata.data.numpy()[i],(28,28)))
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())


for step,(x,y) in enumerate(trainload):

    x=Variable(x.view(-1,28*28))
    # print(x.size())
    # y=Variable(y).type(torch.FloatTensor)
    pred=autoen(x)
    loss = lossfunc(pred, x)
    # print(x.size())
    # print(pred.size())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step %50==0:
        de=autoen(viewdata)
        # print(de.size())
        for i in range(5):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(de.data.numpy()[i],(28,28)))
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.pause(0.1)


plt.ioff()
plt.show()
