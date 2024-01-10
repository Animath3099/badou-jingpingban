import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


"""
基于pytorch框架编写模型训练,实现一个自行构造的十分类任务
规律：x是一个3维向量，预测x的最大值对应的索引为标签的概率, 输出对应3个类别的概率
"""

# 数据
# 随机生成一个3维向量，最大值所在的索引即为分类标签
def build_sample():
    x = np.random.random(3)
    if x[0] > x[1] and x[0] > x[2]:
        return x, 0
    elif x[1] > x[0] and x[1] > x[2]:
        return x, 1
    else:
        return x, 2

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 模型
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)   #w：3 * 10
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)     #w：10 * 3
        self.activation = torch.softmax
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.layer1(x)
        x = self.layer2(x)
        y_pred = self.activation(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred



# 测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    classfication1, classfication2, classfication3 = 0, 0, 0
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            print(y_p, y_t)
            if float(y_p.max(0).indices) == 0 and int(y_t) == 0:
                classfication1 += 1  # 正样本判断正确
            elif float(y_p.max(0).indices) == 1 and int(y_t) == 1:
                classfication2 += 1  # 正样本判断正确
            elif float(y_p.max(0).indices) == 2 and int(y_t) == 2:
                classfication3 += 1  # 正样本判断正确
            else:
                wrong += 1
    correct = classfication1 + classfication2 + classfication3
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 3  # 输入向量维度
    learning_rate = 0.01  # 学习率
    model = TorchModel(input_size, hidden_size1=10, hidden_size2=3)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
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
    torch.save(model.state_dict(), "model1.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 3
    model = TorchModel(input_size, hidden_size1=10, hidden_size2=3)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    #print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        result = [t.numpy() for t in result]
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, float(res), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.61129719, 0.40578238, 0.53190582],
                [0.21879531, 0.21554711, 0.72816835],
                [0.44864318, 0.73564213, 0.84492114],
                [0.90535827, 0.33524455, 0.37741623],
                [0.50122594, 0.12023433, 0.27028237]]

    predict("model1.pt", test_vec)




