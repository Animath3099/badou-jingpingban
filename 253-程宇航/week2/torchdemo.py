import torch
import torch.nn as nn
import numpy as np
import random
import json


# 生成一个样本
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    elif x[0] < x[3]:
        return x, 2
    else:
        return x, 0

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)     # Y接收的值为类似于[[0], [1], [0]]
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 搭建模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.Linear = nn.Linear(input_size, 3)   # 线性层
        # self.activation = torch.sigmoid          # 归一化函数
        self.loss = nn.CrossEntropyLoss()           # 交叉熵损失，内部包含softmax归一化

    # 真实标签时返回loss值，无真实标签时，返回预测值
    def forward(self, x, y=None):
        x = self.Linear(x)
        # y_pred = self.activation(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return torch.softmax(x, dim=1)

# 预测准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        # for y_p, y_t in zip(y_pred, y):
        #     if float(y_p) < 0.5 and int(y_t) == 0:
        #         correct += 1
        #     elif float(y_p) >= 0.5 and int(y_t) == 1:
        #         correct += 1
        #     else:
        #         wrong += 1
        pred_max = torch.argmax(y_pred, dim=1)     # 一个张量中取最大值（dim=1每一行的最大值），输出类似于tensor([1, 0, 2]) （三行），为索引
        correct = (pred_max == y).sum().item()
    print("预测正确的个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num

def main():
    # 参数配置
    epoch_num = 30
    batch_size = 20
    train_sample = 50000
    input_size = 5
    learning_rate = 0.01

    model = TorchModel(input_size)   # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)    # 优化器
    log = []
    train_x, train_y = build_dataset(train_sample)    # 创建训练集
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            if batch_index % 2 == 0:
                optim.step()     # 更新权重
                optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("===========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model4.pth")    # 模型保存

if __name__ == "__main__":
    main()