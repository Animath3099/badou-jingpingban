import torch
import torch.nn as nn
import numpy as np

ce_loss = nn.CrossEntropyLoss()
pred = torch.FloatTensor([[3.2, 1.7, 0.3],
                          [2.2, 0.2, 1.9],
                          [0.5, 5.4, 0.1],
                          [1.4, 1.3, 4.2]])
target = torch.LongTensor([1,0,1,2])
loss = ce_loss(pred, target)
print(loss, "torch输出交叉熵")


def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

#将输入转化为onehot矩阵
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target

#手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    print(f"batch_size:{batch_size}  class_num:{class_num}")  # 4 3
    pred = softmax(pred)
    print("pred:", pred)
    target = to_one_hot(target, pred.shape)
    print("target:", target)
    entropy = - np.sum(target * np.log(pred), axis=1)
    return sum(entropy) / batch_size

print(cross_entropy(pred.numpy(), target.numpy()), "手动实现交叉熵")
