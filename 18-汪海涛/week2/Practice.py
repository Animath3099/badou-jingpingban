# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，根据不同的条件进行三分类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层，输出维度改为3
        self.activation = nn.Softmax(dim=1)  # softmax激活函数，保证输出是概率分布
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失，适用于多分类问题

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            loss = self.loss(x, y)
            return loss
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    print(x)
    if x[0] > x[4]:
        return x, 0
    elif x[1] > x[3]:
        return x, 1
    else:
        return x, 2


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个类别0，%d个类别1，%d个类别2" % (sum(y == 0), sum(y == 1), sum(y == 2)))
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y).sum().item()

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "practice.pt")

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        _, predicted = torch.max(result, 1)
        for vec, res in zip(input_vec, predicted):
            print("输入：%s, 预测类别：%d" % (vec, res.item()))


if __name__ == "__main__":
    main()
    test_vec = [
        [0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
        [0.03963533, 0.5524256, 0.95758807, 0.35520434, 0.24890681],
        [0.01097868, 0.29582528, 0.13625847, 0.34675372, 0.19871392],
        [0.19349776, 0.59416669, 0.92579291, 0.41567412, 0.0012894],
        [0.08797868, 0.89482528, 0.13625847, 0.70075372, 0.39871392],
        [0.19349776, 0.40416669, 0.92579291, 0.41567412, 0.7358894]
    ]
    predict("practice.pt", test_vec)
