import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 设定环境变量（避免某些系统问题）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义模型
class TorchFiveModel(nn.Module):
    def __init__(self, input_size):
        super(TorchFiveModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)# 创建一个具有线性层的神经网络，输入大小为input_size，输出大小为5
        self.softmax = nn.Softmax(dim=1)  # 创建一个Softmax层，用于将输出转换为类别概率

    def forward(self, x):
        output = self.linear(x)  # 将输入传递给线性层
        probabilities = self.softmax(output)  # 使用Softmax将线性输出转换为类别概率
        return probabilities

# 生成一个样本
# def build_sample(): 太过复杂 使用10W个训练样本依旧很低准确率
#     x = np.random.random(10)
#     if x[0] > x[4]:
#         label = 1
#     elif x[0] <= x[4] and x[1] > x[7]:
#         label = 2
#     elif x[0] <= x[4] and x[1] <= x[7] and x[2] > x[6]:
#         label = 3
#     elif x[0] <= x[4] and x[1] <= x[7] and x[2] <= x[6] and x[3] > x[5]:
#         label = 4
#     else:
#         label = 5
#     return x, label

# 下为简单点的5分类任务
def build_sample():
    x = np.random.random(10)  # 随机生成一个10维向量

    if x[0] > 0.8:
        label = 1
    elif x[1] > 0.6:
        label = 2
    elif x[2] > 0.4:
        label = 3
    elif x[3] > 0.2:
        label = 4
    else:
        label = 5

    return x, label


# 生成数据集
def generate_dataset(total_samples_num):
    X, Y = [], []
    for _ in range(total_samples_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y - 1)  # 使标签从 0 开始将标签减1以将标签从1到5映射为0到4
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 评估模型
def evaluate(model, test_sample_num=500):
    model.eval()
    x, y = generate_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1) #选择每个样本中概率最高的类别
        correct = (predicted == y.squeeze().long()).sum().item()
    accuracy = correct / test_sample_num
    return accuracy

# 训练模型
def train_model(model, train_x, train_y, epoch_num, batch_size):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)# 使用Adam优化器进行模型训练
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(len(train_x) // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            output = model(x) #使用模型进行向前传播
            loss = nn.CrossEntropyLoss()(output, y.squeeze().long())  # 使用交叉熵损失计算损失
            loss.backward()  # 反向传播，计算梯度
            optim.step() # 更新模型参数
            optim.zero_grad()  # 清除梯度
            watch_loss.append(loss.item())

        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))]) # 记录准确性和损失
        print(f"Epoch {epoch + 1}: Loss = {np.mean(watch_loss)}, Accuracy = {acc}")

    torch.save({'model_state_dict': model.state_dict()}, 'model.pt') # 保存训练好的模型
    print("Model saved as model.pt")
    return log

# train_model函数用于训练模型，包括前向传播、损失计算、反向传播和参数更新。


# 使用训练好的模型进行预测
def predict(model_path, input_vec, input_size):
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"The file {model_path} does not exist.")
        return

    # 初始化模型
    model = TorchFiveModel(input_size)
    model.eval()

    # 加载模型状态
    checkpoint = torch.load(model_path)
    if 'model_state_dict' not in checkpoint:
        print("Error: 'model_state_dict' not found in the checkpoint.")
        return

    model.load_state_dict(checkpoint['model_state_dict'])

    # 进行预测
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vec)
        result = model(input_tensor)

        for vec, res in zip(input_vec, result):
            predicted_class = torch.argmax(res).item()+1 # 找到概率最高的类别
            print(f"输入：{vec}, 预测类别：{predicted_class}, 概率值：{res[predicted_class]}")

# predict函数用于使用训练好的模型进行预测。
# 主函数
def main():
    train_sample = 1000000
    input_size = 10
    epoch_num = 20
    batch_size = 20

    model = TorchFiveModel(input_size) #初始化模型
    train_x, train_y = generate_dataset(train_sample) #生成训练数据集
    log = train_model(model, train_x, train_y, epoch_num, batch_size)

    plt.plot(range(len(log)), [l[0] for l in log], label="acc") # 绘制准确性曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss") # 绘制损失曲线
    plt.legend()
    plt.show()
    #较难
    # test_vec = [
    #     # 类别 1 的示例：第1个数大于第5个数
    #     [0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #
    #     # 类别 2 的示例：第1个数小于等于第5个数且第2个数大于第8个数
    #     [0.2, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.8, 0.9],
    #
    #     # 类别 3 的示例：第1个数小于等于第5个数且第2个数小于等于第8个数且第3个数大于第7个数
    #     [0.2, 0.1, 0.8, 0.4, 0.5, 0.6, 0.3, 0.7, 0.8, 0.9],
    #
    #     # 类别 4 的示例：第1个数小于等于第5个数且第2个数小于等于第8个数且第3个数小于等于第7个数且第4个数大于第6个数
    #     [0.1, 0.2, 0.3, 0.8, 0.5, 0.4, 0.6, 0.7, 0.8, 0.9],
    #
    #     # 类别 5 的示例：其他情况
    #     [0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6, 0.9, 0.5]
    # ]
    # test_vec = [ #样本
    #     [0.9, 0.5, 0.3, 0.1, 0.7, 0.2, 0.6, 0.4, 0.8, 0.9],  # 类别1
    #     [0.7, 0.7, 0.5, 0.1, 0.2, 0.3, 0.8, 0.4, 0.8, 0.9],  # 类别2
    #     [0.3, 0.8, 0.6, 0.3, 0.4, 0.2, 0.6, 0.7, 0.5, 0.9],  # 类别3
    #     [0.1, 0.3, 0.8, 0.7, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 类别4
    #     [0.5, 0.4, 0.3, 0.6, 0.9, 0.8, 0.7, 0.2, 0.1, 0.1]  # 类别5
    # ]
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 类别1
        [0.2, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.1, 0.8, 0.9],  # 类别2
        [0.3, 0.2, 0.8, 0.4, 0.5, 0.6, 0.3, 0.7, 0.8, 0.9],  # 类别3
        [0.1, 0.2, 0.3, 0.8, 0.5, 0.4, 0.6, 0.7, 0.8, 0.9],  # 类别4
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6, 0.9, 0.5]  # 类别5
    ]

    # if x[0] > 0.8:
    #     label = 1
    # elif x[1] > 0.6:
    #     label = 2
    # elif x[2] > 0.4:
    #     label = 3
    # elif x[3] > 0.2:
    #     label = 4
    # else:
    #     label = 5
    predict("model.pt", test_vec, input_size)

# 运行主函数
if __name__ == "__main__":
    main()
