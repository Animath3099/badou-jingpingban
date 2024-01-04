import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt



class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 10)   # 线性层
        self.relu = nn.ReLU()                    # 激活层
        self.layer2 = nn.Linear(10, 6)            # 线性层
        self.activation = nn.Softmax(dim=-1)         # 归一化函数
        self.loss = nn.CrossEntropyLoss()          # 损失函数
        self.fc = nn.Sequential(self.layer1, self.relu, self.layer2)
    def forward(self, x, y = None):
        y_pred = self.activation(self.fc(x))
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个6维向量，最大数所在索引为样本所在类
def build_sample():
    x = np.random.random(6)
    y = np.argmax(x)
    return x, y

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,  y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个0样本，%d个1样本, %d个2样本，%d个3样本，%d个4样本, %d个5样本" % (torch.flatten(y).tolist().count(0), torch.flatten(y).tolist().count(1), torch.flatten(y).tolist().count(2), torch.flatten(y).tolist().count(3), torch.flatten(y).tolist().count(4), torch.flatten(y).tolist().count(5)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(np.argmax(y_p)) == int(y_t):
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 1000  # 每次训练样本个数
    train_sample = 500000  # 每轮训练总共训练的样本总数
    input_size = 6  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = MyModel(input_size)
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
            x = train_x[batch_size * batch_index : batch_size * (batch_index + 1)]
            y = torch.flatten(train_y[batch_size * batch_index: batch_size * (batch_index + 1)]).long()
            loss = model(x, y)   # 计算损失
            loss.backward()      # 计算梯度
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
    input_size = 6
    model = MyModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, int(np.argmax(res.numpy())), float(np.max(res.numpy()))))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec = [[0.60578631, 0.07441138, 0.57715984, 0.99828556, 0.36192575, 0.2669379 ],
                 [0.36153601, 0.77006066, 0.64059183, 0.87090249, 0.35062505, 0.35061608],
                 [0.33553676, 0.24326617, 0.64603087, 0.90733969, 0.86440603, 0.89816663],
                 [0.96459071, 0.32136263, 0.3548649, 0.17018462, 0.54599781, 0.03686998],
                 [0.88438404, 0.90289996, 0.99811565, 0.58413253, 0.37639151, 0.31870757]]
    predict("model.pt", test_vec)