import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as pyplot

'''
学习任务:在[1,2,3,4,5]类似的五维向量中,判断哪个元素最大并返回其索引值
我认为交叉熵中不存在y的期望预测值,因为在生成数据集时y只代表分类标签属性,所以只存在x的期望预测值
所有x均为5维度tensor,所有y都为1维度tensor
'''

# 制作样本方法(在5维数组中哪个元素最大，返回其索引值并附加在数组后)
def build_sample():
    x = np.random.random(5)         #制作一个0-1之间随机数的5维数组(数据类型为：numpy.ndarray类似于list)
    x_max_value = np.sort(x)[-1]    #取出该数组的最大值
    # print(x)                      #验证结果
    # print(type(x))                #验证结果为numpy.ndarray类型
    # print(x_max_value)            #验证结果
    if x[0] == x_max_value:
        return x, 0
    elif x[1] == x_max_value:
        return x, 1
    elif x[2] == x_max_value:
        return x, 2
    elif x[3] == x_max_value:
        return x, 3
    else:
        return x, 4
# print(build_sample())              #验证结果
# ex1:(array([0.55404593, 0.36983275, 0.0983084 , 0.83673918, 0.29558395]), 3)其中前方array阵列为后面使用的x，后面的int数字为后面使用的y_label
# print(type(build_sample()))        #验证结果为tuple类型

# 按照build_sample方法生成训练样本
def build_dataset(total_sample_num):
    X = []                                                        #存放array阵列
    Y_label = []                                                  #存放阵列最大值索引(由build_sample函数决定)
    for i in range(total_sample_num):
        x, y_label = build_sample()                               #分别将x,y在类似ex1中遍历出array阵列与int类型索引
        X.append(x)                                               #将每次取出的x(array)添加到X列表末尾
        Y_label.append(y_label)                                   #将int类型的y转化成list类型,并添加到Y列表末尾
    # ex2：(tensor([[0.6462, 0.1316, 0.5711, 0.1983, 0.8202]]), tensor([4]))
    return torch.FloatTensor(X), torch.LongTensor(Y_label)        #此时会返回一个前后都为tensor的tuple数据类型

# 设计一个输入维度是5,输出维度是5的线性层(即五分类任务)
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)    #nn.Linear会自动生成一个5*5的矩阵,确保一个1*5的输出结果,即输出了x的期望预测值(x_expect)
        # self.activation = torch.sigmoid                    #使用sigmoid作为激活函数(激活函数:为loss提供曲线因素),使用交叉熵会自动使用softmax作为损失函数
        '''
        该loss计算方法可以类比为
        cross_entropy_loss = nn.CrossEntropyLoss()
        x_expect = torch.FloatTensor([[0.3, 0.1, 0.3, 0.5, 0.7]]) 
        y_label = torch.LongTensor([3]) 
        cross_entropy_loss = cross_entropy_loss(x_expect, y_label)
        '''
        self.loss = nn.functional.cross_entropy              #使用交叉熵的方式计算loss
# 计算loss值
    def forward(self, x, y_label = None):
        x_expect = self.linear(x)
        # y_expect = self.activation(x)
        if y_label is not None:
            return self.loss(x_expect, y_label)
        else:
            return x_expect

# 测试每轮模型准确率
def evaluate(model):
    model.eval()             #模型进入测试模式
    test_sample_num = 100    #样本数量
    x, y_label = build_dataset(test_sample_num)
    # print(x)               #验证结果
    # print(y)               #验证结果
    # torch.sum()是求和用的,在其中加上条件判断,用他来计算每个样本的总和
    count_result_0 = torch.sum(y_label == 0)
    count_result_1 = torch.sum(y_label == 1)
    count_result_2 = torch.sum(y_label == 2)
    count_result_3 = torch.sum(y_label == 3)
    count_result_4 = torch.sum(y_label == 4)
    print('''
    本次预测集中有
    0号样本:{}个
    1号样本:{}个
    2号样本:{}个
    3号样本:{}个
    4号样本:{}个
    '''.format(count_result_0, count_result_1, count_result_2, count_result_3, count_result_4))
    correct, wrong = 0, 0
    with torch.no_grad():                                            # with代表声明状态,这里使用.no_grad是不进行反向传播(即不会计算梯度,也不会有梯度下降过程)
        x_expect = model(x)                                          # 预测期望x值
        for x_exp, y_label_true in zip(x_expect, y_label):
            if torch.argmax(x_exp) == int(y_label_true):             #torch.argmax()是取出该tensor最大值索引用的,torch.max()就是取出该tensor最大值索引用的
                correct += 1
            else:
                wrong += 1
    print('''
    正确预测个数:{}
    错误预测个数:{}
    正确率:{}%
    '''.format(correct, wrong, correct * 100 / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 1000                  #训练轮数
    batch_size = 10                  #每次训练样本个数
    every_train_sample_num = 5000    #每轮训练样本个数
    input_size = 5                   #输入向量维度
    learn_rate = 0.0001               #学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择Adam的优化器(SGD高级版)优化器作用：梯度下降找loss极小值
    # .parameters()是可训练的所有参数(就是线性层中y=wx+b中的w与b)
    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
    log = []                                                               #日志空list
    # 创建由build_dateset函数调用的训练集
    train_x, train_y = build_dataset(every_train_sample_num)               #创建训练集次数的total_sample_num入参被'每轮训练样本个数'赋值
    # 训练过程
    for epoch in range(epoch_num):
        model = model.train()                                              #将模型设置为训练状态
        watch_loss = []
        for batch_index in range(every_train_sample_num // batch_size):    #'//'代表整除,即每轮需要循环5000/10=500次
            # train_x与train_y都是tensor数据类型,可以用类似list索引取值的方式对每个batch_size取一次值(x,y也是tensor类型)
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y_label = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y_label)          #计算loss值
            loss.backward()                   #梯度计算
            optimizer.step()                  #更新优化器权重
            optimizer.zero_grad()             #梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)                 # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        # 保存模型
    torch.save(model.state_dict(), "model_1.pt")
    # 画图
    print(log)
    pyplot.plot(range(len(log)), [l[0] for l in log], label="acc")   # 画acc曲线
    pyplot.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    pyplot.legend()
    pyplot.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))                                  # 加载训练好的权重
    print(model.state_dict())

    model.eval()                                                                   # 测试模式
    with torch.no_grad():                                                          # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))                       # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))    # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.89349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model_1.pt", test_vec)
