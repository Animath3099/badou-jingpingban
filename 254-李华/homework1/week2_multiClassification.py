import torch
import torch.nn as nn
import numpy as np

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''

基于pytorch框架编写模型训练
实现5分类任务，找出最大值
规律：x是一个5维向量，如果向量下标为n的数最大，则为n类数据，一共0到5类数据

'''



# 建立模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred


# 生成一个样本，样本的生成方法，代表我们要学习的规律
# 随机生成一个5维向量，如果第一位上的数字最大，则第一位为第一类
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index


# 随机生成一批样本
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
def evaluate(model):
    model.eval()  # 评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        y_pred = model(x)
        print("y_pred: " + y_pred)



# 训练模型
def train():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  # 训练模式
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model.forward(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
    torch.save(model.state_dict(), "model.pt")


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入:%s, 预测类别:%s, 概率值:%s" % (vec, torch.argmax(res), res))




if __name__ == "__main__":
    train()
    # print(build_sample())

    test_vec = [[0.47846086,0.15599675,0.31034123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84432981],
                [0.78797788,0.62982528,0.13625847,0.34895372,0.09871392],
                [0.89349776,0.59456669,0.92579291,0.41567412,0.7356987]]
    predict("model.pt", test_vec)
