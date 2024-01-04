#coding:utf8

import torch
import numpy 


def Softmax(x):
    res = []
    for data in x:
        res.append(numpy.exp(data))
    res = [r/sum(res) for r in res]
    return res

print(numpy.exp(1))
x = numpy.array([1,2,3,4])
print("pytorch.sofrmax: %s"%torch.softmax(torch.Tensor(x),0))
# print("Softmax {:<4}".format(Softmax(x)))
# print_res = ["{:.4f}".format(item) for item in Softmax(x)]
print_res = ["%.4f" % item for item in Softmax(x)]
print("Softmax"," ".join(print_res))
