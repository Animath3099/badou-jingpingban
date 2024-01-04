#coding:utf8

import torch
import numpy as np
import torch.nn as nn

class TorchModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(TorchModel,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size,output_size)


    def forward(self,x):
        x = self.layer1(x)
        y_pred = self.layer2(x)
        return y_pred

input_data = np.array([
    [3.1, 1.3, 1.2],
    [2.1, 1.3, 13]
])
#这是一个全连接层，只需要输入每层输入输出节点个数就可以
torch_model = TorchModel(3,5,2)
print(torch_model.state_dict())
print('---------------------')
torch_x = torch.FloatTensor(input_data)
y_pred = torch_model(torch_x)
print("torch_model_output: {}".format(y_pred))

print('======================')

torch_model_w1 = torch_model.state_dict()['layer1.weight'].numpy()
torch_model_b1 = torch_model.state_dict()['layer1.bias'].numpy()
torch_model_w2 = torch_model.state_dict()['layer2.weight'].numpy()
torch_model_b2 = torch_model.state_dict()['layer2.bias'].numpy()
#这里取出模型参数，自定义模型，验证结果,自定义模型不需要构建节点，forward函数直接完成计算，因此不需要输入节点数
class MyModel():
    def __init__(self, w1, w2, b1, b2):
        self.b1 = b1
        self.b2 = b2
        self.w1 = w1
        self.w2 = w2

    def forward(self, x):
        hidden_output = np.dot(x,self.w1.T) + self.b1
        output = np.dot(hidden_output,self.w2.T) + self.b2
        #下面这种写法相加的时候会导致无法广播
        # hidden_output = np.dot(self.w1,x.T) + self.b1
        # output = np.dot(self.w2,hidden_output.T) + self.b2
        return y_pred
        
my_model = MyModel(torch_model_w1, torch_model_w2, torch_model_b1, torch_model_b2)
#这里是使用np进行计算，无需flatten，只需要注意矩阵乘法方式就行
y_pred = my_model.forward(np.array(input_data))
print("my_model_output {}".format(y_pred))
        