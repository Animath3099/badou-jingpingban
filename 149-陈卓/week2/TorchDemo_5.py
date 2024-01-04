import numpy as np
import torch.nn as nn
import torch

"""
5分类的任务，输入10维向量，判断属于哪一类
规律：
10个数两两相加，x1+x2,x3+x4,x5+x6,x7+x8,x9+x10，
如果x1+x2最大输出0，...如果x9+x10最大输出4
eg.
[1,2,3,4,5,6,7,8,9,10] 输出 [4]
[1,2,3,40,5,6,7,8,9,10] 输出 [1]
[1,2,3,4,5,60,7,8,9,10] 输出 [2]
"""

# 构建模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5) # 线性层
        # cross_entropy已经包含了softmax，不用激活函数
        self.loss = nn.functional.cross_entropy # loss函数采用交叉熵

    def forward(self, x, y=None):
        x = self.linear(x)
        # 如果输入了y值就返回loss，(模型训练)，如果没有输入y值就返回softmax之后的结果，(用模型预测)
        if y is not None:
            y = y.squeeze()
            return self.loss(x, y)
        else:
            return torch.softmax(x, dim=1)
        

# 创建数据集
def data_create():
    x = np.random.randn(10)
    summed_list = [x[i] + x[i+1] for i in range(0, 10, 2)]
    y = summed_list.index(max(summed_list))
    return x, y

def dataset_create(data_size):
    X = []
    Y = []
    for i in range(data_size):
        x, y = data_create()
        X.append(x)
        Y.append([y])

    X_np = np.array(X)
    Y_np = np.array(Y)

    return torch.FloatTensor(X_np), torch.LongTensor(Y_np)


# 评估模型准确性
def evaluate(model):
    model.eval()
    test_sample_num = 100

    # 创建测试数据集
    test_x, test_y = dataset_create(test_sample_num)

    correct, wrong = 0, 0

    # 进行模型预测
    with torch.no_grad():
        y_pred = model(test_x)  
    
        # 把y_pred的结果转换成输出分类的类别
        _, pred_classes = torch.max(y_pred, dim=1)
        # 把类别转换成列表方便和输入的数据对比
        pred_classes = pred_classes.tolist()
        # 判断结果正确性
        for i in range(test_sample_num):
            if test_y[i] == pred_classes[i]:
                correct += 1
            else:
                wrong += 1
    
    print(f'预测正确的数量：{correct},预测错误的数量：{wrong},正确率：{correct / test_sample_num}')
    return correct / test_sample_num


# 训练模型
def train():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每批训练样本数量
    train_sample = 5000  # 每轮训练样本数量
    input_size = 10  # 输入向量维度
    learning_rate = 0.001  # 学习率

    # 构建模型
    model = TorchModel(input_size)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练数据集
    train_x, train_y = dataset_create(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            # 计算loss
            loss = model.forward(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
        # 评估本轮模型训练之后的效果
        acc = evaluate(model)
    
    return model


# 使用训练好的模型进行预测
def predict(model, input_vec):
    model.eval()
    # 先把input_vec转换成tensor
    x = torch.FloatTensor(input_vec)
    with torch.no_grad():
        # 和在evaluate函数中写的是一样的
        y_pred = model(x)
        print(y_pred)
        _, pred_classes = torch.max(y_pred, dim=1)
        pred_classes = pred_classes.tolist()
    for vec, pred in zip(input_vec, pred_classes):
        print(f'输入的数据是：{vec},预测的分类是:{pred}')


if __name__ == "__main__":
    trained_model = train()
    test_vec = [[-0.43509963,  0.89363365, -0.8778037 , -0.41289946,  0.39377444,
       -0.82315422, -0.89470173, -0.13825848, -0.40048552, -0.91969376],
       [ 0.84479792, -1.67968496,  0.80312764, -1.00128502,  0.92928357,
       -0.60079204,  1.90377213,  0.84687896,  0.06595296,  0.35919071],
       [ 0.29025241,  0.15748136,  0.34093902,  0.10535747, -0.68984946,
        0.41956907,  1.31971683, -0.63423324, -1.01676918, -1.04342063],
        [-0.36030447,  2.26320401, -0.42458291, -0.69424848, 100,
       -1.026434  ,  1.10348548,  0.09926412, -1.33097508,  1.32038728],
       [-0.75357232,  0.43836663, -0.41997688, -1.07992539, -1.37009533,
       -0.83666346, -0.25328421,  0.20687047,  1.10422881, -1.17487159]]
    predict(trained_model, test_vec)

