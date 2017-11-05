import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Dataset


epochs=1
batch_size=64
time_step=28
input_size=28
lr=0.01

traindata=torchvision.datasets.MNIST(
    root='./mnist',
    transform=torchvision.transforms.ToTensor(),
    train=True
)
testdata=torchvision.datasets.MNIST(root='./mnist',train=False)
test_x=Variable(torch.unsqueeze(testdata.test_data,1)).type(torch.FloatTensor)[:2000]/255.
# print(test_x[0])
test_y=testdata.test_labels.numpy().squeeze()[:2000]

trainload=Dataset.DataLoader(traindata,batch_size=batch_size,shuffle=True)


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn1=nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out=nn.Linear(64,10)
    def forward(self, x):
        r_out,(h_n,h_c)=self.rnn1(x,None)
        out=self.out(r_out[:,-1,:])
        return  out
rnn=RNN()
optimizer=torch.optim.Adam(rnn.parameters(),lr=lr)
lossfunc=nn.CrossEntropyLoss()

for epoch in range(epochs):
    for step,(x,y) in enumerate(trainload):

        x=Variable(x.view(-1,28,28))
        # print(x[1])
        # print(x.size())
        y=Variable(y)
        output=rnn(x)
        optimizer.zero_grad()
        loss=lossfunc(output,y)
        loss.backward()
        optimizer.step()
        if step%50==0:
            print('loss:',loss.data[0])

test_pred=rnn(test_x[:10].view(-1,28,28))
# test_pred=torch.max(test_pred,1)[1].type(torch.FloatTensor)
# print(test_pred.size())
# print(test_y.size())
#此处不匹配，需要增加内容
# accuracy=torch.mean((test_pred.data==test_y).type(torch.FloatTensor))

# print('accu:',accuracy)
# print('ten predictions:',test_pred.data.numpy()[:10])
# print('ten originalS:',test_y.numpy()[:10])

pred_y=torch.max(test_pred,1)[1].data.numpy().squeeze()
print(pred_y,'predition')
print(test_y[:10],'real number')


