import torch
import numpy as np

x = np.random.random((4,5))
print(x)
bn = torch.nn.BatchNorm1d(5)
print("bn:", bn)
y_bn = bn(torch.from_numpy(x).float())
print("torch y_bn:", y_bn)
print("=============1================")
print(bn.state_dict())
print(bn.state_dict()["weight"])
print(bn.state_dict()["bias"])
print(bn.state_dict()["running_mean"])
print(bn.state_dict()["running_var"])
print(bn.state_dict()["num_batches_tracked"])
print("=============2================")
gamma = bn.state_dict()["weight"].numpy()
beta = bn.state_dict()["bias"].numpy()
num_features = 5
eps = 1e-05
momentum = 0.1

#gamma = np.random.randn(num_features)
#beta = np.random.randn(num_features)
print("gamma:", gamma)
print("beta:", beta)
running_mean = np.zeros(num_features)
running_var = np.zeros(num_features)
mean = np.mean(x, axis=0)
var = np.var(x, axis=0)
print("mean:", mean)
print("var:", var)
print("=============3================")
running_mean = momentum * running_mean + (1-momentum) * mean
running_var = momentum * running_var + (1-momentum) * var
print("running_mean:", running_mean)
print("running_var:", running_var)
print("=============4================")
x_norm = (x-mean) / np.sqrt(var+eps)
y = gamma * x_norm + beta
print("ours y:", y)