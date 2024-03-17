#coding:utf8
import numpy as np
import torch
import torch.nn as nn

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN,self).__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)

    def forward(self,x):
        x = self.layer(x)
        return x


class MyDiyRnn(nn.Module):
    def __init__(self, ih_weight, hh_weight, hidden_size):
        self.ih_weight = ih_weight
        self.hh_weight = hh_weight
        self.hidden_size = hidden_size

    def forward(self, x):
        ht = np.zeros((self.hidden_size))#这里ht的shape为4*1
        res = []
        for xt in x:#这里xt为3*1
            rest_1 = np.dot(self.ih_weight,xt)#这里为4*3 * 3*1 -> 4*1
            rest_2 = np.dot(self.hh_weight,ht)#这里为4*4 * 4*1 -> 4*1
            rest = np.tanh(rest_1 + rest_2)
            ht = rest
            res.append(rest)
        return np.array(res), ht
hidden_size = 4
input_size = 3
Rnn_model = TorchRNN(input_size, hidden_size)
ih_weight = Rnn_model.state_dict()['layer.weight_ih_l0']#这里shape为4*3
hh_weight = Rnn_model.state_dict()['layer.weight_hh_l0']#这里shape为4*4
print(ih_weight, ih_weight.shape)
print(hh_weight, hh_weight.shape)

x = np.array([
        [1, 2, 3],
        [3, 4, 5],
        [5, 6, 7]
        ])#这里shape为3*3
x = np.array(x)
torch_x = torch.FloatTensor([x])

output, h_out = Rnn_model(torch_x)
#print(h_out)
print(output.detach().numpy(), 'torch模型预测结果')
print(h_out.detach().numpy(),'torch模型预测隐含层结果')
print('__________________________________________________')
Diy_model = MyDiyRnn(ih_weight, hh_weight, hidden_size)
output_diy, h_diy = Diy_model.forward(x)
print('diy模型预测结果:', output_diy)
print('diy模型隐藏结果:', h_diy)





