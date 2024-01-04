# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, out_features):
        super(TorchModel, self).__init__()
        self.linear_01 = nn.Linear(input_size, 16)  # 线性层
        self.relu_01 = nn.ReLU()
        self.linear_02 = nn.Linear(16, 8)
        self.relu_02 = nn.ReLU()
        self.linear_03 = nn.Linear(8, out_features)
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear_01(x)
        x = self.relu_01(x)
        x = self.linear_02(x)
        x = self.relu_02(x)
        y_pred = self.linear_03(x)

        if y is not None:
            return self.loss(y_pred, y) # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num, n_features, n_informative, n_redundant, n_classes):
    X, Y = make_classification(n_samples=total_sample_num, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_classes=n_classes, shuffle=True)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, test_x, test_y, n_classes):
    model.eval()
    test_sample_num = test_x.shape[0]

    print("測試集數據分布")
    for i in range(n_classes):
        print(f"類別{i}: {len(test_y[test_y == i])}")

    correct = 0
    with torch.no_grad():
        y_pred = model(test_x)  # 模型预测
        _, predicted = torch.max(y_pred.data, dim=1)
        correct = (predicted == test_y).sum().item()

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 50  # 每次训练样本个数
    dataset_size = 10000
    feature_size = 5  # 输入向量维度
    informative_feature_size = 4
    redundant_feature_size = feature_size - informative_feature_size
    n_classes = 3
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(feature_size, n_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    X, Y = build_dataset(dataset_size, feature_size, informative_feature_size, redundant_feature_size, n_classes)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1, random_state=42)

    train_sample = train_x.shape[0]

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, test_x, test_y, n_classes)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_multi_class.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    n_classes = 3
    model = TorchModel(input_size, n_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        _, result = torch.max(result.data, dim=1)
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 1.03504317, 0.18920843],
                [-0.94963533, -0.5524256, 2.95758807, 0.95520434, 0.84890681],
                [0.78797868, 3.67482528, 0.13625847, -0.34675372, -1.19871392],
                [2.19349776, 0.59416669, -1.92579291, 0.41567412, -4.7358894]]
    predict("model_multi_class.pt", test_vec)
