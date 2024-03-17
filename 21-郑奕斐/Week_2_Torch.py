#Setting Module
import torch
import torch.nn as nn # Neural Network 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np # Transfer data
import pandas as pd #Loading data from csv
import seaborn as sns # Data Visualization
import matplotlib.pyplot as plt # Data Visualization
from sklearn.model_selection import train_test_split #Split dataset

#Original Data from https://tianchi.aliyun.com/dataset/dataDetail?dataId=92968
#Request - Building a Multiple Classification Task
 
# 1) Loading Data
data = pd.read_csv('Desktop/forestfires.csv')
data['area'].describe()
#Accoring to Data distribution, in this task, choice item - area as Multiple Classification(True - y)
# Low fire area - area < 5 ----- 0
# Middle fire area - 10 > area > 5 ----- 1
# High fire area - area > 10 ----- 2

# 2）Preparing Data

df = pd.read_csv('Desktop/forestfires.csv')
def classification_data(row):
    if row['area'] > 10:
        return 2
    elif row['area'] > 5 and row['area'] < 10:
        return 1
    else:
        return 0

df['Fire disaster damage degree'] = df.apply(classification_data,axis=1)
df.drop('month', axis=1, inplace=True)
df.drop('day', axis=1, inplace=True)
df.drop('area',axis=1,inplace=True)
df.to_csv('forest.csv',index=False)

df = pd.read_csv('forest.csv')
feature_columns = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
features = df[feature_columns].values

input_data = np.array(features)

true_y = df['Fire disaster damage degree'].values
X_train, X_test, y_train, y_test = train_test_split(input_data, true_y, test_size=0.2, random_state=42)

def create_dataset(X, y, batch_size):
    input_data = torch.tensor(X, dtype = torch.float32)
    true_y = torch.tensor(y, dtype = torch.float32)
    dataset = TensorDataset(input_data, true_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

batch_size = 40
train_dataset = create_dataset(X_train, y_train, batch_size)
test_dataset = create_dataset(X_test, y_test, batch_size)

#Neural Network Building
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)
        self.activation = nn.ReLU()
    
    def forward(self, x, y = None):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            loss = nn.functional.cross_entropy(y_pred, y)
            return loss
        else:
            return y_pred
        

#Evaluation
def evaluate(model):
    model.eval()
    x, y = input_data, true_y
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 40  # 每次训练样本个数
    input_size = 10  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    X_train, X_test, y_train, y_test = train_test_split(input_data, true_y, test_size=0.2, random_state=42)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(0, len(X_train), batch_size):    
            x = torch.tensor(X_train[batch_index * batch_size : (batch_index + 1) * batch_size]).clone().detach()
            y = torch.tensor(y_train[batch_index * batch_size : (batch_index + 1) * batch_size]).clone().detach()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)# 计算loss 
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
    print(log)
    log_df = pd.DataFrame(log, columns=["Accuracy", "Loss"])
    sns.lineplot(data=log_df, markers=True)
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_data):
    input_size = 10
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32))  # 将输入数据转换为张量，并进行模型预测
        _, predicted = torch.max(y_pred, 2)
        correct = (predicted == torch.tensor(y_test, dtype=torch.long)).sum().item()
        total = y_test.shape[0]
        acc = correct / total
        return acc

if __name__ == "__main__":
    main()
