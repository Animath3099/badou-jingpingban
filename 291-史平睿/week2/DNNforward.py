import torch
import torch.nn as nn
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) # w:3 * 5
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # w:5 * 2 

    def forward(self, x):
        x = self.layer1(x)
        print(x.shape) # 2 * 5
        y_pred = self.layer2(x)
        print(y_pred.shape) # 2 * 2
        return y_pred

class DiyModel:
    def __init__(self, w1,b1, w2,b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        hidden = np.dot(x, self.w1.T)+self.b1
        y_pred = np.dot(hidden, self.w2.T)+self.b2
        return y_pred

x = np.array([[3.1, 1.3, 1.2],
             [2.1, 1.3, 13]])

torch_model = TorchModel(3,5,2)
print(torch_model.state_dict())
print("---------------------------")
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_b1 = torch_model.state_dict()["layer1.bias"].numpy()
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_b2 = torch_model.state_dict()["layer2.bias"].numpy()
print(torch_model_w1, "torch w1 权重")
print(torch_model_b1, "torch b1 权重")
print("---------------------------")
print(torch_model_w2, "torch w2 权重")
print(torch_model_b2, "torch b2 权重")
print("---------------------------")
torch_x = torch.FloatTensor(x)
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：", y_pred)

diy_model = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
y_pred_diy = diy_model.forward(np.array(x))
print("diy模型预测结果：", y_pred_diy)