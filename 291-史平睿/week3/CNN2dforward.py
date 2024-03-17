import torch
import torch.nn as nn
import numpy as np

class TorchCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super().__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, kernel, bias=False)

    def forward(self, x):
        return self.layer(x)

class DiyModel:
    def __init__(self, input_height, input_width, weights, kernel_size):
        self.height = input_height
        self.width = input_width
        self.weights = weights
        self.kernel_size = kernel_size

    def forward(self, x):
        output = []
        for kernel_weight in self.weights:
            kernel_weight = kernel_weight.squeeze().numpy()
            kernel_output = np.zeros((self.height - kernel_size + 1, self.width - kernel_size +1))
            for i in range(self.height - kernel_size + 1):
                for j in range(self.width - kernel_size + 1):
                    window = x[i:i+kernel_size, j:j+kernel_size]
                    kernel_output[i, j] = np.sum(kernel_weight * window)
            output.append(kernel_output)
        return np.array(output)

in_channel = 1
out_channel = 3
kernel_size = 2
torch_model = TorchCNN(in_channel, out_channel, kernel_size)
#print(torch_model.state_dict())
torch_w = torch_model.state_dict()["layer.weight"]
print(torch_w, torch_w.numpy().shape)  #(3,1,2,2)

x = np.array([[0.1,  0.2,  0.3,  0.4],
              [-3,   -4,   -5,   -6 ],
              [5.1,  6.2,  7.3,  8.4],
              [-0.7, -0.8, -0.9, -1 ],
              [2.2,  3.3,  4.4,  5.5]])
torch_x = torch.FloatTensor([[x]])
output = torch_model.forward(torch_x)
output = output.detach().numpy()
print(output, output.shape, "torch模型预测结果\n")
print("-----------------------------")
print(x.shape)  #(5,4)
print(x.shape[0])
print(x.shape[1])
diy_model = DiyModel(x.shape[0], x.shape[1], torch_w, kernel_size)
output = diy_model.forward(x)
print(output, output.shape, "diy模型预测结果")