# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

'''
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务

规律: 
x 是一个 5 维向量, 
如果第一个值大于第五个值, 且第二个值大于第四个值, 则标签为 0,
如果第一个值大于第五个值, 且第二个值小于等于第四个值, 则标签为 1,
如果第一个值小于等于第五个值, 且第二个值大于第四个值, 则标签为 2,
如果第一个值小于等于第五个值, 且第二个值小于等于第四个值, 则标签为 3

模型: x -> linear -> softmax -> y
损失函数: 交叉熵损失
输入: 5 维向量
输出: 4 分类


'''

num_classes = 4  # 分类数

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)    # 线性层
        self.activation = torch.softmax  # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x, dim=1)  # (batch_size, num_classes) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
        
# 随机生成一个5维向量, 
# 如果第一个值大于第五个值, 且第二个值大于第四个值, 则标签为0,
# 如果第一个值大于第五个值, 且第二个值小于等于第四个值, 则标签为1,
# 如果第一个值小于等于第五个值, 且第二个值大于第四个值, 则标签为2,
# 如果第一个值小于等于第五个值, 且第二个值小于等于第四个值, 则标签为3
def build_sample():
    x =np.random.random(5)
    if x[0] > x[4]:
        if x[1] > x[3]:
            return x, 0
        else:
            return x, 1
    else:
        if x[1] > x[3]:
            return x, 2
        else:
            return x, 3
        
# 随机生成一批样本
# 各种样本均匀生成
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
    model.eval()
    X, Y = build_dataset(1000)
    label_set = set(Y.numpy())
    print('本次预测集中共有%d个类别' % len(label_set))
    for label in label_set:
        print('\t类别%d共有%d个样本' % (label, sum(Y.numpy() == label)))
    Y_pred = model(X)
    Y_pred = torch.argmax(Y_pred, dim=1)
    acc = torch.sum(Y_pred == Y) / Y.shape[0]
    print('正确预测个数: %d' % torch.sum(Y_pred == Y).item())
    print('准确率: %f' % acc.item())
    return acc.item()

# 训练代码
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size)

    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(int(train_sample / batch_size)):
            X, Y = build_dataset(batch_size)
            loss = model(X, Y)  # 计算损失
            optimizer.zero_grad()   # 梯度清零
            loss.backward()     # 计算梯度
            optimizer.step()    # 更新权重
            optimizer.zero_grad()   # 梯度清零
            watch_loss.append(loss.item())  # 记录loss值
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)   # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), 'model_muti.pt')
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, torch.argmax(res).item(), torch.max(res).item()))


if __name__ == '__main__':
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model_muti.pt", test_vec)




