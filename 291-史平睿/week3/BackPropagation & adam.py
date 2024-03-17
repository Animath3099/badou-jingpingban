import torch
import torch.nn as nn
import numpy as np
import copy

class TorchModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = nn.Linear(hidden_size, hidden_size, bias = False)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        y_pred = self.layer(x)
        y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

class DiyModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        x = np.dot(x, self.weight.T)
        y_pred = self.diy_sigmoid(x)
        if y is not None:
            return self.diy_mse_loss(y_pred, y)
        else:
            return y_pred

    def diy_sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def diy_mse_loss(self, y_pred, y_true):
        return np.sum(np.square(y_pred-y_true)) / len(y_pred)

    def calculate_grad(self, y_pred, y_true, x):
        grad_mse = 2/len(x)*(y_pred-y_true)
        grad_sigmoid = y_pred*(1-y_pred)
        grad_w11 = grad_mse[0] * grad_sigmoid[0] * x[0]
        grad_w12 = grad_mse[1] * grad_sigmoid[1] * x[0]
        grad_w21 = grad_mse[0] * grad_sigmoid[0] * x[1]
        grad_w22 = grad_mse[1] * grad_sigmoid[1] * x[1]
        grad = np.array([[grad_w11, grad_w12],
                         [grad_w21, grad_w22]])
        return grad.T

def diy_sgd(grad, weight, learning_rate):
    return weight - learning_rate * grad

def diy_adam(grad, weight):
    alpha = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = 0
    mt = 0
    vt = 0
    t = t+1
    gt = grad
    mt = beta1 * mt + (1-beta1) * gt
    vt = beta2 * vt + (1-beta2) * gt ** 2
    mth = mt / (1 - beta1 ** t)
    vth = vt / (1 - beta2 ** t)
    weight = weight - (alpha * mth/(np.sqrt(vth) + eps))
    return weight

x = np.array([-0.5, 0.1])
y = np.array([0.1, 0.2])

torch_model = TorchModel(2)
torch_model_w = torch_model.state_dict()["layer.weight"]
print(torch_model_w, "初始化权重")

torch_x = torch.from_numpy(x).float().unsqueeze(0)
torch_y = torch.from_numpy(y).float().unsqueeze(0)
torch_loss = torch_model.forward(torch_x, torch_y)
print("torch模型计算loss:", torch_loss)

numpy_model_w = copy.deepcopy(torch_model_w.numpy())
diy_model = DiyModel(numpy_model_w)
diy_loss = diy_model.forward(x, y)
print("diy模型计算loss:", diy_loss)

torch_loss.backward()
print(torch_model.layer.weight.grad, "torch计算梯度")

grad = diy_model.calculate_grad(diy_model.forward(x), y, x)
print(grad, "diy计算梯度")

print("========================================")
# 设定优化器
learning_rate = 0.001
#optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(torch_model.parameters())
# torch梯度更新
optimizer.step()

update_torch_model_w = torch_model.state_dict()["layer.weight"]
print(update_torch_model_w, "torch更新后权重")

#diy_update_w = diy_sgd(grad, numpy_model_w, learning_rate)
diy_update_w = diy_adam(grad, numpy_model_w)
print(diy_update_w, "diy更新权重")