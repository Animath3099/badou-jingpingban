import random
import sys

import numpy as np
import torch.nn as nn
import torch

# 建立Torch类
class MySelfTorchModel(nn.Module):
    def __init__(self, input_dim, out_dim):
        # 固定写法
        super(MySelfTorchModel, self).__init__()
        # 定义4个线性层和2个激活函数
        self.linear1 = nn.Linear(input_dim,128)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128,128)
        self.linear4 = nn.Linear(128,out_dim)
        self.softmax = torch.softmax
        self.softmax2 = torch.softmax
        self.relu = torch.relu
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y_true=None):
        # 前馈计算
        y_pre = x
        y_pre = self.linear1(y_pre)
        y_pre = self.softmax2(y_pre, dim=1)
        y_pre = self.linear2(y_pre)
        y_pre = self.relu(y_pre)
        y_pre = self.linear3(y_pre)
        y_pre = self.linear4(y_pre)
        y_pre = self.softmax(y_pre, dim=1)
        # 如果没有穿真实y，就认为是在计算预测值，否则认为是在训练，计算loss
        if y_true is None:
            return y_pre
        return self.loss(y_pre,y_true)

def to_onehot(ture_index):
    y = np.zeros(8)
    y[ture_index % 8] = 1
    return y

def get_data(count):
    """
    构造数据  将x向量每个元素都加和，0-1属于第一类，1-2属于第二类，以此类推。进行八分类
    :param count: 数据量
    :return: 返回的数据
    """
    X, Y = [], []
    for i in range(count):
        x = np.random.rand(8)
        X.append(x)
        # 将x加和，进行0-7的八分类
        Y.append(to_onehot(int(np.sum(x))))
    # return X,Y # 注意传入torch的x和y的数据类型转化
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def test_right(y_pre, y):
    # 判断当前预测和真实值是否正确
    return y_pre.argmax() == y.argmax()



def evaluate(model):
    right_num, error_num = 0, 0
    model.eval()
    # 创造测试集
    X,Y = get_data(100)
    with torch.no_grad():
        Y_pre = model(X)
        # 对比测试结果
        for y_pre, y in zip(Y_pre, Y):
            if test_right(y_pre, y):
                right_num +=1
            else:
                error_num +=1
    print(f"正确预测个数：{right_num}, 正确率：{(right_num/(right_num+error_num))*100}%")


model_path = "model.pth"
def main():
    # 超参数定义
    epoch_num = 100 # 训练多少轮
    batch_size = 20 # 每个批次训练多少数据
    data_count = 1000 # 样本数据构造多少
    lr = 1e-4 # 学习率
    suss_loss = 1e-5 # 提前推出学习的loss

    # 创建训练数据
    X, Y = get_data(data_count)
    # 创建模型对象
    model = MySelfTorchModel(8, 8)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),lr)

    # 进行训练
    for epoch in range(epoch_num):
        # 开启训练模型
        model.train()
        # 当前轮的loss
        loss_epoch = 0
        for batch in range(data_count//batch_size):
            # 取出当前批次的数据
            X_batch = X[batch * batch_size : (1 + batch) * batch_size]
            Y_batch = Y[batch * batch_size : (1 + batch) * batch_size]
            # 当前批次的loss
            loss_batch = model(X_batch,Y_batch)
            # 计算梯度
            loss_batch.backward()
            # 反向传播
            optim.step()
            # 梯度归零
            optim.zero_grad()
            # 将批次的loss计入本轮loss
            loss_epoch += loss_batch.item()
        # 输出本轮结果
        loss_epoch = loss_epoch / ((data_count // batch_size) + 1)
        print(f"第{epoch}轮loss：{loss_epoch}")
        # 进行模型测试
        evaluate(model)
        if loss_epoch <= suss_loss:
            break
    # 保存模型
    torch.save(model.state_dict(), model_path)
    return


def predict(model_path, test_vec):
    # 加载模型
    model = MySelfTorchModel(8, 8)
    model.load_state_dict(torch.load(model_path))
    # print(model.state_dict())
    # 计算结果
    model.eval()
    with torch.no_grad():
        res_list = model(torch.FloatTensor(test_vec))
    for x, res in zip(test_vec,res_list):
        print(f"输入结果：{np.sum(x)}, 预测结果：第{res.argmax()}类, 预测概率：{res[res.argmax()]}")


if __name__ == "__main__":
    main()
    # 构造预测数据
    # test_vec = [[0.93223407,0.40740507,0.86567478,0.51157166,0.99664116,0.16382238,0.51289429,0.82419898],
    #              [0.26531475,0.74943657,0.68805544,0.06415783,0.45772746,0.83999211,0.66245318,0.38585432],
    #              [0.54457643,0.72456975,0.84895514,0.73215419,0.7271725,0.07878114,0.28918756,0.02815741],
    #              [0.32758733,0.19697205,0.35245708,0.25930026,0.12579894,0.97787087,0.29025962,0.65674961],
    #              [0.25676195,0.76948399,0.40519595,0.16281787,0.18955977,0.69246735,0.33739367,0.66706586],
    #              [0.09610491,0.59295526,0.1118053,0.11848185,0.8918123,0.33950702,0.92872811,0.73823808],
    #              [0.06675149,0.79316063,0.80320048,0.85208456,0.88968092,0.57641858,0.22449592,0.03738898],
    #              [0.96353019,0.20634115,0.23831531,0.38415818,0.32650253,0.22049587,0.30925843,0.02518939]]
    # # 进行预测
    # predict(model_path, test_vec)
    sys.exit(0)
