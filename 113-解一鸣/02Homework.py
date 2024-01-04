"""

规律：x是一个10维向量，第1个数与第2个数相加，第3个与第4个相加，
若第1个数与第2个数相加的值最大，输出 0，第3个与第4个相加值最大，输出 1，以此类推
输出的样本空间为：0，1，2，3，4

"""
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


# 构建模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.linear(x)
        # 输入y值就返回loss
        if y is not None:
            y = y.squeeze()
            return self.loss(x, y)
        else:
            # 没有输入 y，返回softmax
            return torch.softmax(x, dim=1)


def build_sample():
    # 构建单条样本数据
    # 生成一个有10个数字的数组
    random_numbers = np.random.uniform(0, 100, 10)
    # 计算元素的和
    sums = random_numbers[::2] + random_numbers[1::2]
    # 找到和最大的索引
    max_sum_index = np.argmax(sums)
    return random_numbers, max_sum_index


def build_sample_set(total_sample_num):
    # 构建全部样本数据
    x_sample = []
    y_sample = []
    for i in range(total_sample_num):
        x, y = build_sample()
        x_sample.append(x)
        # y的样本空间为：0，1，2，3，4
        y_sample.append([y])
    return torch.FloatTensor(np.array(x_sample)), torch.LongTensor(np.array(y_sample))


def statistical(result):
    # 输出各个元素数量的情况
    np_result = np.array(result)
    (elements, elements_nums) = np.unique(np_result, return_counts=True)
    for element, count in zip(elements, elements_nums):
        print(f"{element}: {count}")


def evaluate(model):
    model.eval()
    total_sample_num = 100
    x_sample, y_sample = build_sample_set(total_sample_num)
    flatten_y_sample = np.array(y_sample).flatten()
    with torch.no_grad():
        y_pred = model(x_sample)
        # 使用 torch.max 在每一行中找到最大值及其索引，即最可能情况对应的索引
        output_pred = np.array(torch.max(y_pred, dim=1)[-1])
        correct_totals = np.sum(output_pred == flatten_y_sample)
        wrong_totals = total_sample_num - correct_totals
        print(f"预测正确总计：{correct_totals}，预测错误总计：{wrong_totals}, 正确率：{correct_totals / total_sample_num}")


def start_train():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 10  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练集，正常任务是读取训练集
    x_sample, y_sample = build_sample_set(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        # 训练模式
        model.train()
        for batch_index in range(train_sample // batch_size):
            x = x_sample[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = y_sample[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 计算loss
            loss = model.forward(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            evaluate(model)

    # 保存模型
    torch.save(model.state_dict(), "model.pt")


# 使用训练好的模型做预测
def predict(model_path, x_sample):
    model = TorchModel(10)
    model.load_state_dict(torch.load(model_path))
    # 测试模式
    model.eval()
    with torch.no_grad():
        y_pred = model(x_sample)
        output_pred = np.array(torch.max(y_pred, dim=1)[-1])
        for x, output in zip(x_sample, output_pred):
            print(f"输入的数据：{x}，预测结果：{output}")


if __name__ == "__main__":
    start_train()
    predict("model.pt", build_sample_set(10)[0])
