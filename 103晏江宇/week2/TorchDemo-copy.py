# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

'''
基于pytorch框架编写模型训练
实现一个自行构建的找规律（机器学习）任务
规律：x是一个5维向量，如果第一个数>第五个数，则为正样本，反之为负样本

'''

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层
        self.activation = torch.sigmoid     # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x) # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred,y) # 预测值和真实值计算损失
        else:
            return y_pred   # 输出预测结果

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个6维向量，如果第1个值大于第5个值，且第1个值大于第6个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(6)
    if x[0] > x[4] and x[0] > x[5]:
        return x,1
    else:
        return x,0


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X),torch.FloatTensor(Y)
# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset((test_sample_num))
    print("本次预测集中共有%d个正样本，%d个负样本" %(sum(y),test_sample_num - sum(y)))
    correct,wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1 # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1 # 正样本判断正确
            else:
                wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct,correct / (correct + wrong)))
        return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 1220      # 训练轮数
    batch_size = 20     # 每次训练样本个数
    train_sample = 5000     # 每轮训练总的训练样本总数
    imput_size = 6  # 输入向量维度
    learning_rate = 0.001   #学习率
    # 建立模型
    model = TorchModel(imput_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()    # 更新权重
            optim.zero_grad()   # 梯度归零
            watch_loss.append(loss.item())
        print("===============\n第%d轮平均loss:%f" % (epoch + 1,np.mean(watch_loss)))
        acc = evaluate(model) # 测试本轮模型结果
        log.append([acc,float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')   # 画acc曲线
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 6
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()    # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))    # 模型预测
    for vec, res in zip(input_vec,result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))     # 打印结果


if __name__ == "__main__":
    main()
    dataset = build_dataset(4)
    # print(dataset)
    test_vec = [[0.4400, 0.7205, 0.4749, 0.5302, 0.4539, 0.4325],   # 0
                [0.2617, 0.6900, 0.2992, 0.0806, 0.0004, 0.2550],   # 1
                [0.7745, 0.8150, 0.9581, 0.0601, 0.3366, 0.7233],   # 1
                [0.9107, 0.9127, 0.6484, 0.9225, 0.6017, 0.4939]]   # 1
    predict("model.pt", test_vec)
