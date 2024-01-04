# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的多分类任务
规律：x是一个5维向量，如果第1个数大于第5个数且小于等于其他数，则为第1类；如果第1个数大于第4个数且小于等于其他数，则为第2类；其他情况则为第3类。

"""


class TorchModel(nn.Module): # TorchModel 类是一个 PyTorch 模型，继承自 nn.Module。通过调用 super(TorchModel, self).__init__()，你在 TorchModel 的构造函数中调用了父类 nn.Module 的构造函数，确保了 TorchModel 实例能够正确地初始化父类的属性。
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()  #表示调用 TorchModel 类的父类的构造函数，即调用 nn.Module 类的构造函数。这是使用 Python 类继承时的一种常见用法，目的是在子类的构造函数中调用父类的构造函数，以确保子类继承了父类的属性和方法。
        self.linear = nn.Linear(input_size, num_classes)  # 线性层

    def forward(self, x):
        # x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)，这里不需要activation的原因：If you are using CrossEntropyLoss, you typically do not need to apply a sigmoid activation explicitly because CrossEntropyLoss includes a softmax function internally. Ensure that the model's output size matches the number of classes in your classification task.

        # # The calculation of the loss typically occurs outside of the forward method, specifically in the training loop when you compare the model's output with the ground truth labels.
        # if y is not None: # 当输入真实标签，返回loss值；无真实标签，返回预测值
        #     return self.loss(x, y)  # 预测值和真实值计算损失
        # else:
        #     return x  # 输出预测结果
        return self.linear(x)

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第1个数大于第5个数且小于等于其他数，则为第1类；如果第1个数大于第4个数且小于等于其他数，则为第2类；其他情况则为第3类
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4] and all(x[0] <= val for val in x[1:4]):
        return x, 0 # Class 1
    elif x[0] > x[3] and x[0] <= x[4] and all(x[0] <= val for val in x[1:3]):
        return x, 1 # Class 2
    else:
        return x, 2 # Class 3


# 随机生成一批样本
# 第1，2，3类样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y) # In PyTorch, when dealing with classification tasks, it's a common practice to use torch.LongTensor for labels (Y) and to organize your labels as a 1D tensor. Therefore, you would typically use Y.append(y) rather than Y.append([y]).
        X_array = np.array(X)
        Y_array = np.array(Y)

        # Convert NumPy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X_array)
        Y_tensor = torch.LongTensor(Y_array)

    return X_tensor, Y_tensor

# Training function
def train(model, criterion, optimizer, train_x, train_y, epochs, batch_size):
    log = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0

        # Iterate over batches
        for batch_start in range(0, len(train_x), batch_size):
            batch_end = batch_start + batch_size
            x_batch = train_x[batch_start:batch_end]
            y_batch = train_y[batch_start:batch_end]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate the number of correct predictions
            _, predicted_classes = torch.max(outputs, 1)
            correct_predictions += (predicted_classes == y_batch).sum().item() # .item() extracts the count as a Python scalar

        # Calculate accuracy for the epoch
        accuracy = correct_predictions / len(train_x)

        # Print average loss for the epoch
        average_loss = running_loss / (len(train_x) / batch_size)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

        # Save accuracy and loss for plotting
        log.append({'epoch': epoch + 1, 'loss': average_loss, 'accuracy': accuracy})

    return log

# Evaluation function
def evaluate(model, test_x):
    model.eval()
    with torch.no_grad():
        output = model(test_x)
        _, predicted_classes = torch.max(output, 1)
        # # Example output tensor (2D tensor)
        # output = torch.tensor([[0.8, 0.2, 0.1],
        #                        [0.2, 0.5, 0.9],
        #                        [0.3, 0.4, 0.6]])
        #
        # # Find the predicted classes
        # _, predicted_classes = torch.max(output, 1)
        #
        # print(predicted_classes.numpy())  # Output: [0, 2, 2]
    return predicted_classes.numpy()


def main():
    # 配置参数
    input_size = 5  # 输入向量维度
    num_classes = 3 # 输出的类别数
    batch_size = 20  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    learning_rate = 0.001  # 学习率
    epochs = 50  # 训练轮数


    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Loss函数
    criterion = nn.CrossEntropyLoss()

    # Build dataset
    train_x, train_y = build_dataset(train_sample)

    # Train the model
    log = train(model, criterion, optimizer, train_x, train_y, epochs, batch_size)

    # 保存模型
    torch.save(model.state_dict(), "../Week2HOMEWORK/model.pt")
    # 画图
    epochs_list = [entry['epoch'] for entry in log]
    loss_list = [entry['loss'] for entry in log]
    accuracy_list = [entry['accuracy'] for entry in log]

    plt.plot(epochs_list, loss_list, label='Loss')
    plt.plot(epochs_list, accuracy_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()