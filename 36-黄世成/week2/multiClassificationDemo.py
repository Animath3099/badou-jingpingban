# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
问题：根据用户资产情况（总资产，近3个月入金金额，近3个月出金金额，近1年交易笔数，近1年交易量），
     对用户进行分类（0-低风险，1-中风险，2-高风险）
"""


class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128,3),
            nn.Softmax()
        )
        self.loss = nn.functional.cross_entropy

    def forward(self, x):
        return self.net(x)



# 生成一个样本，总共5个维度，每个维度的随机值都设置一个范围
# 根据预定的规律构建Y
def build_sample():
    v1 = np.random.uniform(200, 2000)
    v2 = np.random.uniform(10, 1000)
    v3 = np.random.uniform(10, 1000)
    v4 = np.random.uniform(0, 1000)
    v5 = np.random.uniform(0, 100000)

    x = np.array([v1, v2, v3, v4, v5])

    # 低风险客户
    if (x[0] > 1000):
        if (x[1] > x[2] and x[3] > 100 and x[4] > 10000):
            return x, 0
        else:
            return x, 1
    
    # 中风险客户
    elif (500 <= x[0] <= 1000):
        if (x[1] >= x[2] and 50 <= x[3] <= 100 and 5000 <= x[4] <= 10000):
            return x, 1
        else:
            return x, 2
    
    # 高风险客户
    else:
        return x, 2


def normalize(x, min=None, max=None):
    if min is None and max is None:
        min = x.min(axis=0)
        max = x.max(axis=0)
        print("min: %f, max: %f",min, max)
    x_norm = (x - min) / (max - min)
    return x_norm

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    X_norm = normalize(np.array(X),min=MIN,max=MAX)
    Y = np.array(Y)

    return torch.FloatTensor(X_norm), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    correct, wrong = 0, 0
    with torch.no_grad():
        outputs = model(x)  # 模型预测
        result = torch.softmax(outputs.data, 1)
        _, y_pred = torch.max(result, 1)
        # print(y_pred)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print("lable: %d, pred: %d", y_t, y_p)
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

MIN = np.array([200,10,10,0,0])
MAX = np.array([2000,1000,1000,1000,100000])
def main():
     # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = MultiClassficationModel(input_size)
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
            result = model(x)
            loss = model.loss(result, y)
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
    # print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        normalize_input_vec = normalize(np.array(input_vec),min=MIN, max=MAX) 
        result = model.forward(torch.FloatTensor(normalize_input_vec))  # 模型预测
        prod, pred = torch.max(result, 1)
    for vec, res, prd in zip(input_vec, pred, prod):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, res, prd))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[1300, 30, 20, 110, 12000],
                [800, 20, 30, 100, 0],
                [800, 70, 30, 600, 6000],
                [300, 30, 50, 200, 2300]]
    predict("model.pt", test_vec)
