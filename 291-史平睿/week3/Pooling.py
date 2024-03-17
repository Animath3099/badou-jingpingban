import torch
import torch.nn as nn

layer = nn.AvgPool1d(4)

x = torch.rand([3,4,5])
print(x)
print(x.shape)
x = x.transpose(1,2)
print(x.shape, "½»»»ºó")
y = layer(x)
print(y)
print(y.shape)
y = y.squeeze()
print(y)
print(y.shape)