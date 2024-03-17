import torch
import torch.nn as nn
import numpy as np


kernel_size = 3
input_dim = 5
hidden_size = 4
torch_cnn1d = nn.Conv1d(input_dim, hidden_size, kernel_size)
#print(torch_cnn1d.state_dict())
print(torch_cnn1d.state_dict()["weight"])
print(torch_cnn1d.state_dict()["bias"])
print("-----------------------")
for key, weight in torch_cnn1d.state_dict().items():
    print(key, weight.shape)

def numpy_cnn1d(x, state_dict):
    weight = state_dict["weight"].numpy()
    bias = state_dict["bias"].numpy()
    sequence_output = []
    for i in range(0, x.shape[1] - kernel_size + 1):
        window = x[:, i:i+kernel_size]
        kernel_outputs = []
        for kernel in weight:
            kernel_outputs.append(np.sum(kernel * window))
        sequence_output.append(np.array(kernel_outputs) + bias)
    return np.array(sequence_output).T

print("-----------------------")
#x = torch.from_numpy(np.random.random((4, input_dim)))
x = np.random.random((4, input_dim))
x = x.transpose(1, 0)
print("x:", x, x.shape)
torch_x = torch.Tensor([x])
print("torch_x:", torch_x, torch_x.shape)
print(torch_cnn1d(torch_x))
print(numpy_cnn1d(x, torch_cnn1d.state_dict()))