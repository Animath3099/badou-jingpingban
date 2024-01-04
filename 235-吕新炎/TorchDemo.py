# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
"""
基于pytorch框架编写模型训练
实现一个自行构造的多分类任务
规律：x是一个5维向量，根据数值范围划分为三个类别
"""
class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层
        self.activation = nn.Softmax(dim=1)  # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, output_size)
        y_pred = self.activation(x)  # (batch_size, output_size) -> (batch_size, output_size)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，根据数值范围划分为三个类别
def build_sample():
    x = np.random.random(5)
    value_sum = sum(x)
    if value_sum < 1.5:
        return x, 0  # 类别0
    elif value_sum < 2.5:
        return x, 1  # 类别1
    else:
        return x, 2  # 类别2
# 随机生成一批样本
# 三个类别均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)
# 测试代码
# 用来测试每轮模型的准确率
def predict(model_path, input_vec):
    model_path.eval()
    x, y = build_dataset(input_vec)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model_path(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)  # 获取最大概率对应的类别
        correct += (predicted == y).sum().item()  # 统计预测正确的数量
    acc = correct / input_vec
    print("本次预测集中共有%d个样本，正确预测个数：%d, 正确率：%f" % (input_vec, correct, acc))
    return acc
def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size = 3  # 输出类别数量
    learning_rate = 0.001  # 学习率
    model = TorchModel(input_size, output_size)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    log = []
    train_x, train_y = build_dataset(train_sample)  # 创建训练集
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
        acc = predict(model, 1000)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        torch.save(model.state_dict(), "model.pt")  # 保存模型
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc") # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return
if __name__ == "__main__":
     # main()
     output_size = 3  # 替换为你的模型输出维度
     model = TorchModel(5, output_size)  # 输入_size 和 output_size 应该是你的模型输入和输出维度
     model.load_state_dict(torch.load("D:/NLP/20231228/zuoye/model.pt"))
     print(model.state_dict())

     # 定义测试数据
     test_vec = [
         [0.4723, 0.8654, 0.1235, 0.6789, 0.3456],
         [0.7890, 0.2345, 0.5678, 0.1234, 0.9012],
         [0.1234, 0.5678, 0.9012, 0.3456, 0.7890],
         [0.3456, 0.7890, 0.1234, 0.5678, 0.9012],
         [0.5678, 0.9012, 0.3456, 0.7890, 0.1234],
         [0.7890, 0.1234, 0.5678, 0.9012, 0.3456],
         [0.9012, 0.3456, 0.7890, 0.1234, 0.5678],
         [0.1234, 0.5678, 0.9012, 0.3456, 0.7890],
         [0.3456, 0.7890, 0.1234, 0.5678, 0.9012],
         [0.5678, 0.9012, 0.3456, 0.7890, 0.1234]
     ]

     # 转换为 PyTorch 张量
     test_vec = torch.FloatTensor(test_vec)

     correct, total = 0, len(test_vec)

     # 遍历测试数据集并进行预测
     for i in range(total):
         x = test_vec[i].unsqueeze(0)  # 添加一个 batch 维度
         y_pred = model(x)
         _, predicted = torch.max(y_pred, 1)  # 获取最大概率对应的类别

         # 假设你有真实的标签，这里以 build_sample() 函数生成的标签为例
         true_label = build_sample()[1]

         if predicted.item() == true_label:
             correct += 1

     acc = correct / total
     print("预测准确率：", acc)