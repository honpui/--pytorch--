import torch
import torchvision
import torch.utils.data as Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

batch_size=64
lr=0.0001
n_ideas=5
art_components=15
paint_points=np.vstack([np.linspace(-1,1,art_components) for _ in range(batch_size)])

def artist_works():
    a=np.random.uniform(1,2,size=batch_size).reshape(-1,1)
    paintings=a*np.power(paint_points,2)+(a-1)
    paintings=torch.from_numpy(paintings).float()
    return Variable(paintings)

class GAN_gen(torch.nn.Module):
    def __init__(self):
        super(GAN_gen,self).__init__()
        self.generator=nn.Sequential(
            nn.Linear(5,128),
            nn.ReLU(),
            nn.Linear(128,art_components)

        )

    def forward(self, x):
        gen=self.generator(x)
        return  gen

class GAN_dis(nn.Module):
    def __init__(self):
        super(GAN_dis,self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(art_components, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        dis_x = self.discriminator(x)
        return dis_x

gen=GAN_gen()
dis=GAN_dis()
optimizer_gen=torch.optim.Adam(gen.parameters(),lr=lr)
optimizer_dis=torch.optim.Adam(dis.parameters(),lr=lr)

plt.ion()
for i in range(10000):
    artist_paintings=artist_works()
    g_idea=Variable(torch.randn(batch_size,n_ideas))
    g_paintings=gen(g_idea)
    prob0=dis(artist_paintings)
    prob1=dis(g_paintings)

    loss_d=-torch.mean(torch.log(prob0)+torch.log(1-prob1))
    loss_g=torch.mean(torch.log(1-prob1))


    optimizer_dis.zero_grad()
    loss_d.backward(retain_graph=True)
    optimizer_dis.step()

    loss_g.backward()
    optimizer_gen.step()
    optimizer_gen.zero_grad()




    if i %30==0:
        plt.cla()
        plt.plot(paint_points[0],g_paintings.data.numpy()[0])
        plt.plot(paint_points[0],2*np.power(paint_points[0],2)+1)
        plt.plot(paint_points[0],1*np.power(paint_points[0],2)+0)
        plt.pause(0.1)

plt.ioff()
plt.show()