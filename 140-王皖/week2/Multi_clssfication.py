# coding:utf8

#任务：实现输入数据的四分类,根据输入数据各元素和的大小进行分类，每个元素在0~1之间
# 0 < sum(x) <= 0.5 , y = 1
# 0.5 < sum(x) <= 1 , y = 2
# 1 < sum(x) <= 1.5 , y = 3
# 1.5 < sum(x) <= 2 , y = 4
# 2 < sum(x) <= 2.5 , y = 5
# 其他， y=0

#0、导入环境包
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


#1、构建模型框架：模型深度、激活函数、损失函数、神经元数量...
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 6)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用均方差损失， 计算预测输出与真实输出的距离

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        # print(y_pred.shape, y.shape)
        # print(y_pred, y)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


#2、定义函数,获取样本数据,
# 0 < sum(x) <= 0.5 , y = 1
# 0.5 < sum(x) <= 1 , y = 2
# 1 < sum(x) <= 1.5 , y = 3
# 1.5 < sum(x) <= 2 , y = 4
# 2 < sum(x) <= 2.5 , y = 5
# 其他， y=0
def build_database(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x = np.random.random(5)
        temp = sum(x)
        if 0 < temp <= 0.5:
            X.append(np.array(x))
            Y.append(1)
        elif 0.5 < temp <= 1:
            X.append(np.array(x))
            Y.append(2)
        elif 1 < temp <= 1.5:
            X.append(np.array(x))
            Y.append(3)
        elif 1.5 < temp <= 2:
            X.append(np.array(x))
            Y.append(4)
        elif 2 < temp <= 2.5:
            X.append(np.array(x))
            Y.append(5)
        else:
            X.append(np.array(x))
            Y.append(0)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))

#3、定义准确率函数（用于验证模型）
def evaluate(model):
    model.eval() #将模型设置成测试模式
    test_sample_num = 100
    x, y = build_database(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测等价于 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p).item() == int(y_t):
                correct += 1  # 样本判断正确
            else:
                wrong += 1

            # 方法二：
            # y_p = nn.functional.softmax(y_p, dim=0)
            # class_id = np.argmax(y_p)
            # if class_id == int(y_t):
            #     correct += 1
            # else:
            #     wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

#4、定义训练函数
def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器, 用于进行梯度更新，寻找最小值。
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_database(train_sample)
    # 训练过程
    print("=============================训练开始===============================")
    for epoch in range(epoch_num):
        model.train() #将模型设置成训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss, 等价于model.forward(x, y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果，用于验证
        log.append([acc, float(np.mean(watch_loss))])
    print("=============================训练结束！===============================")
    # 保存模型
    torch.save(model.state_dict(), "model_6class.pth")
    # 画图
    fig = plt.figure(figsize=(10,8))
    plt.title("The Performance of Model")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend(['acc', 'loss'])
    plt.show()
    plt.savefig('./plot.png', dpi=300, bbox_inches = 'tight')
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，等价于model（x）
    for vec, res in zip(input_vec, result):
        res = nn.functional.softmax(res, dim=0) #模型预测结果有大于1或者小于0 的情况，通过softmax()函数转换成0-1之间概率分布
        class_id = np.argmax(res) #找出模型预测结果概率值最大对应的索引，即类别
        print("输入：%s, 和为：%f, 预测类别：%d, 概率值：%f" % (vec, sum(vec), class_id, res[class_id]))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894],
                [0.28797868,0.07482528,0.13625847,0.34675372,0.19871392],
                [0.01349776,0.29416669,0.02579291,0.01567412,0.0358894]]
    predict("model_6class.pth", test_vec)
