import torch 
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
实现一个自行构建的找规律（基于pytorch框架编写机器学习模型训练任务）
规律：x是一个5维向量，如果第1个数最大，输出[1,0,0,0,0]；第二个数最大，输出[0,1,0,0,0]；
第三个数最大，输出[0,0,1,0,0]；第四个数最大，输出[0,0,0,1,0]；第五个数最大，输出[0,0,0,0,1]
"""
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层 5分类任务
        self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 随机生成一个5维向量，如果第一个值最大，则生成[1,0,0,0,0]，以此类推
def build_sample():
    x = np.random.random(5)
    if x[0] > x[1] and x[0] > x[2] and x[0] > x[3] and x[0] >  x[4]:
        return x,[1,0,0,0,0]
    elif x[1] > x[0] and x[1] > x[2] and x[1] > x[3] and x[1] > x[4]:
        return x,[0,1,0,0,0]
    elif x[2] > x[0] and x[2] > x[1] and x[2] > x[3] and x[2] > x[4]:
        return x,[0,0,1,0,0]
    elif x[3] > x[0] and x[3] > x[1] and x[3] > x[2] and x[3] > x[4]:
        return x,[0,0,0,1,0]
    else:
        return x,[0,0,0,0,1]


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    #print("x:", x)
    #print("y:", y)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)
        #print("y_pred:", y_pred)
        for y_p, y_t in zip(y_pred, y):
            #print("y_p:", y_p)
            #print("y_t:", y_t)
            #print("argmax(y_p):", np.argmax(y_p))
            #print("argmax(y_t):", np.argmax(y_t))
            if np.argmax(y_p) == np.argmax(y_t) and float(y_p[np.argmax(y_p)]) > 0.5:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)  # 创建训练集，正常任务是读取训练集
    for epoch in range(epoch_num):  # 训练过程
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size : (batch_index+1)*batch_size]
            loss = model.forward(x, y)  # 输入真实值和标签，计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 梯度更新
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("============\n第%d轮平均loss:%f" % (epoch+1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.pt")  # 保存模型

    print(log)  # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入：%s，输入类别：%s；预测类别：%s，概率值：%s" % (vec, np.argmax(vec), np.argmax(res), res))
        ## 存储四舍五入后的子列表
        #rounded_sublist = []
        ## 遍历子列表中的每个元素并四舍五入
        #for i in range(len(res)):
        #    rounded_sublist.append(round(res[i]))  # 四舍五入后的列表
        #print("输入：%s，预测类别：%s，概率值：%s" % (vec, rounded_sublist, res))


if __name__ == "__main__":
    main()
    #test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #        [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #        [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #        [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    #predict("model.pt", test_vec)
