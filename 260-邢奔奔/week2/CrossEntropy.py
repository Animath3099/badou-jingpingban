#coding:utf8

import torch 
import torch.nn as nn
import numpy as np

torch_ce = nn.CrossEntropyLoss()

pred = [
    [0.3, 0.1, 0.3],
    [0.9, 0.2, 0.9],
    [0.5, 0.4, 0.2]
]
pred = torch.FloatTensor(pred)

target = torch.LongTensor([1, 2, 0])
loss = torch_ce(pred,target)
print("torch.CrossEntropy:",loss)

# def one_hot_code(x):
#     leng = max(x) + 1
#     res = []
#     for i in x:
#         tmpres = np.zeros(leng)
#         tmpres[i] = 1
#         res.append(tmpres)
#     return res
#该one hot 编码方式也可以实现，只是要注意索引不能越界,下面这种onehot编码更好
def to_one_hot(target,shape):
    res = np.zeros(shape)
    for i, data in enumerate(target):
        res[i][data] = 1
    return res

def Softmax(x):
    return np.exp(x) /np.sum(np.exp(x),axis=1,keepdims=True)


def CrossEntropyself(y_pred,y_real):
    y_real = to_one_hot(y_real,y_pred.shape)
    y_pred = Softmax(y_pred)
    entropy = -np.sum(y_real * np.log(y_pred),axis=1)
    # return sum(entropy) / len(entropy)
    return np.mean(entropy)

res = CrossEntropyself(pred.numpy(),target.numpy())
print("CorssEntropyself {:.4f}".format(res))