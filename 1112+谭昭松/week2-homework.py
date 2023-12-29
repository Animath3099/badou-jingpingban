# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线形层
        self.activation = nn.Softmax(dim=1)  # Softmax用于多类分类
        self.loss = nn.CrossEntropyLoss()  # CrossEntropyLoss 多分类交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            # 将y重塑为交叉熵损失的一维张量
            return self.loss(y_pred, y.view(-1))   
        else:
            return y_pred
        
# 随机生成一个5维向量
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)  # 使用最大值的索引作为类标签
    return x, label


# 随机生成一批样本，5分类
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
    model.eval() #
    test_sample_num = 500 # 测试样本数量
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad(): # 不需要梯度计算
        y_pred = model(x) # 获取测试数据集的模型预测

        # 通过找到最大值的索引来确定每个样本的预测类别
        y_pred_class = torch.argmax(y_pred, dim=1)

        # 计算正确预测的数量
        correct += torch.sum(y_pred_class == y).item()
        # 计算预测错误的数量
        wrong += torch.sum(y_pred_class != y).item()

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    num_classes = 5  # 多分类的数目
    epoch_num = 21  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.005  # 学习率
    

    # 建立模型
    model = TorchModel(input_size, num_classes)
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
            loss = model(x, y) # 计算loss
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() #  # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model) # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.pt")

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        result_class = torch.argmax(result, dim=1)
    for vec, res in zip(input_vec, result_class):
        print("输入：%s, 预测类别：%d" % (vec, res))

if __name__ == "__main__":
    main()
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.19349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894],
                [0.07889086, 0.15229675, 0.31082123, 0.3504317, 0.18920843],
                [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.8920843],
                [0.07889086, 0.5229675, 0.31082123, 0.03504317, 0.18920843]]
    predict("model.pt", test_vec)



