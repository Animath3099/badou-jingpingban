import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import logging

def logger_config(log_path,logging_name):
    """
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    """
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

logger = logger_config(log_path='train.log',logging_name='multi-classification')


class MultiClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassificationModel, self).__init__()
        self.linear = nn.Linear(input_size, 10)  # 输入为input_size维向量，输出为10维向量的线性层
        self.loss = nn.functional.cross_entropy  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        """
        前向传播
        :param x: 输入样本
        :param y: 输入真实标签
        :return: 当没有真实标签输入时，返回loss值，否则输出预测值
        """
        y_pred = self.linear(x)  # 根据给定x输出预测值
        if y is not None:
            return self.loss(y_pred, y)  # 计算损失函数
        return y_pred  # 推理模式只输出预测值

def build_sample():
    """
    生成一个样本
    随机生成一个10维向量，标签为向量中最大标量的下标
    :return: 返回一个样本和一个真实标签
    """
    x = np.random.random(10)
    max_id = np.argmax(x)
    return x, max_id

def build_dataset(total: int):
    """
    随机构建给定数目的数据集
    :param total: 样本总数
    :return: 返回包含所有样本的向量，和包含所有标签的向量
    """
    X = []
    Y = []
    for i in range(total):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model: MultiClassificationModel):
    """
    评估模型预测正确率
    :param model: 多分类模型
    :return: 预测的正确率
    """
    model.eval()  # 使用评估模式
    test_total_num = 200  # 测试集总数
    X, Y = build_dataset(test_total_num)  # 构建测试集
    correct_num = 0
    with torch.no_grad():  # 推理无需计算梯度
        y_pred = model(X)  # 预测
        for y_pred, y_true in zip(y_pred, Y):
            if torch.argmax(y_pred) == int(y_true):  # 与真实标签进行对比
                correct_num += 1  # 正确个数++

    accuracy = correct_num / test_total_num  # 计算正确率
    logger.info(f'正确预测个数：{correct_num}，正确率：{accuracy}')
    return accuracy

def log_display(train_log):
    """
    训练日志画图
    :param train_log: 训练日志
    :return:
    """
    print(train_log)
    plt.xlabel('epoch')
    plt.plot(range(len(train_log)), [l[1] for l in train_log], label='loss')  # 画loss曲线
    plt.plot(range(len(train_log)), [l[0] for l in train_log], label='acc')  # 画acc曲线
    plt.legend()
    plt.savefig('train.jpg')
    plt.show()

def train(model, epoch_num: int, train_dataset_x, train_dataset_y, batch_size: int, input_size: int, optimizer, model_save_path: str):
    """
    训练模型
    :param model: 模型
    :param epoch_num: 总轮数
    :param train_dataset_x: 训练集样本
    :param train_dataset_y: 训练集真实标签
    :param batch_size: 批量大小
    :param input_size: 输入维度
    :param optimizer: 优化器
    :param model_save_path: 模型保存路径
    :return:
    """
    train_log = []
    for epoch in range(epoch_num):
        model.train()  # 使用训练模式
        train_loss = []

        dataset_total_num = len(train_dataset_x)
        for batch_index in range(dataset_total_num // batch_size):
            # 最后一次可能不足一个batch
            # 取出样本
            x = train_dataset_x[batch_index * batch_size: min((batch_index + 1) * batch_size, dataset_total_num)]
            # 取出标签
            y = train_dataset_y[batch_index * batch_size: min((batch_index + 1) * batch_size, dataset_total_num)]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零

            train_loss.append(loss.item())  # 记录loss

        mean_loss = np.mean(train_loss)  # 计算本轮平均loss

        logger.info(f'=========\n第{epoch + 1}轮平均loss:{np.mean(train_loss)}')
        accuracy = evaluate(model)  # 计算本轮模型预测正确率
        train_log.append([accuracy, mean_loss])

    torch.save(model.state_dict(), model_save_path)  # 保存模型

    log_display(train_log)  # 画图

def predict(model_path, input_vector):
    """
    模型推理
    :param model_path: 模型路径
    :param input_vector: 输入向量
    :return:
    """
    input_size = 10
    model = MultiClassificationModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 使用评估模式
    with torch.no_grad():  # 推理无需计算梯度
        result = model(torch.FloatTensor(input_vector))  # 模型预测
    for vector, res in zip(input_vector, result):
        print(f'输入：{vector}，预测类别: {torch.argmax(res)}，概率值：{res}')

def main():
    # 训练参数
    epoch_num = 200
    batch_size = 20
    train_sample_num = 5000
    input_size = 10
    learning_rate = 0.001

    # 建模
    model = MultiClassificationModel(input_size)

    # 选择Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建数据集
    train_x, train_y = build_dataset(train_sample_num)

    # 训练模型
    train(model=model,
          epoch_num=epoch_num,
          train_dataset_x=train_x,
          train_dataset_y=train_y,
          batch_size=batch_size,
          input_size=input_size,
          optimizer=optimizer,
          model_save_path='model.pt')


if __name__ == "__main__":
    main()

    # 构造测试数据
    test_vector, label = build_dataset(10)
    print(test_vector)

    # 预测
    predict('model.pt', test_vector)
