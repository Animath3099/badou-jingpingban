import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math as m
import json
import matplotlib.pyplot as plt

"""
三分类
输入数据6维
softmax激活函数 输出[x, y, z]，x + y + z = 1
使用交叉熵损失函数趋近分布:[x, y, z] * 正确标签.t() > 0.5 则预测正确
"""


class TorchModelMC(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 3)
        self.activation = torch.softmax
        self.loss = F.cross_entropy

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pre = self.activation(x, 1)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre


# 生成数据集
"""
第一个数比最后一个数小的话 --> 差
第一个数比最后一个数大且第三位数大于等于1.5 --> 好
第一个数比最后一个数大且第三位数小于1.5 --> 一般
"""


def generate_data_sample():
    sample = np.random.uniform(1, 3, size=6)
    if sample[0] > sample[-1]:
        if sample[2] >= 1.5:
            label = [1, 0, 0]
        else:
            label = [0, 1, 0]
    else:
        label = [0, 0, 1]
    return sample, label


def generate_data_set(n):
    X = []
    Y = []
    for i in range(n):
        x, y = generate_data_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_size = 100
    x, y = generate_data_set(test_sample_size)
    good = y.tolist().count([1, 0, 0])
    ave = y.tolist().count([0, 1, 0])
    bad = y.tolist().count([0, 0, 1])
    print("测试样本中有%d个好瓜， %d个差瓜， %d个一般瓜" % (good, bad, ave))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            y_t = y_t.t()
            dot = torch.matmul(y_p, y_t)
            if dot >= 0.5:
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def model_train():
    epoch_num = 1000
    batch_size = 30
    train_sample = 5000
    input_size = 6
    lr = 0.001
    model = TorchModelMC(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    train_x, train_y = generate_data_set(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watchLoss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watchLoss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watchLoss)))

        acc = evaluate(model)
        log.append([acc, float(np.mean(watchLoss))])

    torch.save(model.state_dict(), "gba.pt")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()


# date_set = generate_data_set(10)
# for e in date_set:
#     print(e)


if __name__ == '__main__':
    model_train()
    test_input = [[2.2, 2.3, 2.1, 3.0, 2.3, 1.5]]
    input_size = 6
    model_test = TorchModelMC(input_size)
    model_test.load_state_dict(torch.load("gba.pt"))
    model_test.eval()
    with torch.no_grad():
        result = model_test.forward(torch.FloatTensor(test_input))
        print(result)
    # evaluate(m)
