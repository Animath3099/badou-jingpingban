# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
随机生成5维数组的数据集，每个数组最小值所在索引，即为样本分类标签
"""
class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层
        self.activation = torch.nn.Sigmoid()  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数交叉熵，用于多分类

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最小值所在的索引为样本分类
def build_sample(classification_nums=5):
    x = np.random.random(classification_nums)
    return x, np.argmin(x)


def gt_first_element_count(arr, classification_nums=5):
    cnt = 0
    for idx in range(1, classification_nums):
        if arr[0] > arr[idx]:
            cnt += 1
    return cnt


# 随机生成一批样本
def build_dataset(total_sample_num, classification_nums=5):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(classification_nums)
        X.append(x)
        one_hot = np.zeros(classification_nums)
        one_hot[y] = 1
        Y.append(one_hot)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 统计生成数据的类型分布
def count_sample_classification(sample_tags):
    # 初始化类型分布统计字典
    distribution = {}
    for key in range(5):
        distribution[key] = 0
    # 统计类型分布
    for one_hot in sample_tags:
        distribution[np.argmax(np.array(one_hot))] += 1
    return distribution


def print_sample_distribution(distribution):
    print("本次预测集中，样本分布为")
    print("{:<8}".format("样本类型"), end='')
    for header in distribution.keys():
        print("{:<4}".format(header), end='')
    print()
    print("{:<8}".format("样本数量"), end='')
    for val in distribution.values():
        print("{:<4}".format(val), end='')
    print()


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    counter = count_sample_classification(y)
    print_sample_distribution(counter)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_t[np.argmax(y_p)] == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size = 5
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, output_size)
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
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
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
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 实际类别：%d, 预测类别：%d, 预测结果：%s, 预测输出向量：%s" % (
            vec, np.argmin(np.array(vec)), np.argmax(np.array(res)),
            np.argmin(np.array(vec)) == np.argmax(np.array(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [np.random.random(5) for _ in range(50)]
    predict("model.pt", test_vec)
