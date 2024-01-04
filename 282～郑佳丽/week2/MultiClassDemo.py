# coding:utf-8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于 pytorch框架编写模型训练
实现一个5分类任务，返回最大值索引的下标
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数使用交叉熵

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    # print(x)
    index = np.argmax(x)
    return x, index


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
    print("本次预测集中共有 %d 个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 预测模型
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def train():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算 loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=======\n第 %d 轮平均 loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    #  保存模型
    torch.save(model.state_dict(), "mutil_class_model.pt")
    #  画图
    print(log)
    plt.plot(range(len(log)), [i[0] for i in log], label="acc")
    plt.plot(range(len(log)), [i[1] for i in log], label="loss")
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print(("输入：%s, 预测类别：%d, 概率值：%f" % (vec, int(torch.argmax(res)), res[torch.argmax(res)])))


if __name__ == '__main__':
    # build_sample()
    # train()
    test_vec = [
        [0.1, 0.3, 0.5, 0.01, 0.66],
        [0.13, 0.33, 0.55, 0.01, 0.06],
        [0.01, 0.30, 0.05, 0.01, 0.06],
        [0.17, 0.03, 0.5, 0.91, 0.06],
                ]
    predict("mutil_class_model.pt", test_vec)
