import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 创建样本
def make_sample(a, b):
    x = np.random.random(size=(a, b))
    y = []
    for i in x:
        index = np.argmax(i)
        a = [0] * 3
        a[index] = 1
        y.append(a)

    return torch.FloatTensor(x), torch.FloatTensor(y)


class ClassifyNet(nn.Module):

    # 设置模型结构
    def __init__(self, input_size, out_size):
        super(ClassifyNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 10)
        self.layer2 = nn.Linear(10, out_size)


    # 返回预测值或者loss
    def forward(self, x, y=None):
        x = self.layer1(x)
        x = F.sigmoid(x)
        pre = self.layer2(x)
        if y is not None:
            return F.mse_loss(pre, y)
        else:
            return pre



def evaluate(net):
    # 模式切换为测试，框架内部会做处理
    net.eval()
    x, y = make_sample(100, 3)
    success, wrong = 0, 0
    with torch.no_grad():
        pre_y = net(x)
        for r_a, p_a in zip(y, pre_y):
            p_a = F.softmax(p_a)
            if np.argmax(r_a) == np.argmax(p_a):
                success += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (success, success / (success + wrong)))
    return success / (success + wrong)




def train():
    # 训练轮次
    epoch_num = 20
    # 每批数量
    batch_size = 20
    # 输入维度
    input_size = 3
    out_size = 3
    # 生成样本数量
    sample_size = 5000
    # 学习率 0.001
    lr = 1e-3
    net = ClassifyNet(input_size, out_size)
    # 选择优化器
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    log = []
    for epoch in range(epoch_num):
        net.train()
        x, y = make_sample(sample_size, input_size)
        watch_loss = []
        for i in range(sample_size // batch_size):
            train_x = x[i * batch_size: (i + 1) * batch_size]
            train_y = y[i * batch_size: (i + 1) * batch_size]
            # 计算loss
            loss = net(train_x, train_y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 测试本轮模型结果
        acc = evaluate(net)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(net.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return



def predict(model_path, matrix):
    net = ClassifyNet(3, 3)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    with torch.no_grad():
        result = net(torch.FloatTensor(matrix))
    result = F.softmax(result).detach().numpy()
    ans = [np.argmax(i) for i in result]
    print(ans)



if __name__ == "__main__":
    # x, y = make_sample(10, 3)
    # print(x)
    # print(y)
    train()
    m = np.random.randint(0, 10, (10, 3))
    print(m)
    predict("model.pt", m)
    # [0, 1, 1, 1, 2, 2, 1, 1, 1, 2]










