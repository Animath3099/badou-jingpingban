```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出层节点数量为类别数
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)  # 使用交叉熵损失函数
        else:
            return F.softmax(x, dim=1)  # 输出预测概率

def build_sample():
    x = np.random.random(5)
    # 将类别数量修改为3，如果第一个值大于第五个值，则为类别1；如果第二个值大于第五个值，则为类别2；其他为类别0
    if x[0] > x[4]:
        return x, 1
    elif x[1] > x[4]:
        return x, 2
    else:
        return x, 0

def build_dataset(total_sample_num, num_classes):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 使用 LongTensor 表示类别索引

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, num_classes=3)
    print("本次预测集中共有%d个类别0，%d个类别1，%d个类别2" % (sum(y == 0), sum(y == 1), sum(y == 2)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted_labels = torch.max(y_pred, 1)
        correct = torch.sum(predicted_labels == y).item()
        wrong = len(y) - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 3  # 修改为3个类别
    learning_rate = 0.001
    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample, num_classes)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model_multi_class.pt")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    predict("model.pt", test_vec)
```

    /var/folders/vk/c6_mn1hd72x8nm30y52hjz540000gn/T/ipykernel_76137/3497016100.py:40: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:264.)
      return torch.FloatTensor(X), torch.LongTensor(Y)  # 使用 LongTensor 表示类别索引


    =========
    第1轮平均loss:1.041973
    本次预测集中共有38个类别0，50个类别1，12个类别2
    正确预测个数：56, 正确率：0.560000
    =========
    第2轮平均loss:0.938998
    本次预测集中共有29个类别0，49个类别1，22个类别2
    正确预测个数：58, 正确率：0.580000
    =========
    第3轮平均loss:0.878343
    本次预测集中共有31个类别0，56个类别1，13个类别2
    正确预测个数：70, 正确率：0.700000
    =========
    第4轮平均loss:0.826644
    本次预测集中共有37个类别0，45个类别1，18个类别2
    正确预测个数：67, 正确率：0.670000
    =========
    第5轮平均loss:0.780781
    本次预测集中共有31个类别0，54个类别1，15个类别2
    正确预测个数：78, 正确率：0.780000
    =========
    第6轮平均loss:0.739834
    本次预测集中共有44个类别0，42个类别1，14个类别2
    正确预测个数：72, 正确率：0.720000
    =========
    第7轮平均loss:0.703163
    本次预测集中共有29个类别0，60个类别1，11个类别2
    正确预测个数：84, 正确率：0.840000
    =========
    第8轮平均loss:0.670228
    本次预测集中共有26个类别0，63个类别1，11个类别2
    正确预测个数：87, 正确率：0.870000
    =========
    第9轮平均loss:0.640557
    本次预测集中共有26个类别0，56个类别1，18个类别2
    正确预测个数：79, 正确率：0.790000
    =========
    第10轮平均loss:0.613744
    本次预测集中共有33个类别0，53个类别1，14个类别2
    正确预测个数：86, 正确率：0.860000
    =========
    第11轮平均loss:0.589437
    本次预测集中共有24个类别0，60个类别1，16个类别2
    正确预测个数：80, 正确率：0.800000
    =========
    第12轮平均loss:0.567335
    本次预测集中共有26个类别0，57个类别1，17个类别2
    正确预测个数：85, 正确率：0.850000
    =========
    第13轮平均loss:0.547176
    本次预测集中共有43个类别0，47个类别1，10个类别2
    正确预测个数：91, 正确率：0.910000
    =========
    第14轮平均loss:0.528738
    本次预测集中共有32个类别0，47个类别1，21个类别2
    正确预测个数：85, 正确率：0.850000
    =========
    第15轮平均loss:0.511827
    本次预测集中共有36个类别0，50个类别1，14个类别2
    正确预测个数：89, 正确率：0.890000
    =========
    第16轮平均loss:0.496278
    本次预测集中共有24个类别0，55个类别1，21个类别2
    正确预测个数：84, 正确率：0.840000
    =========
    第17轮平均loss:0.481946
    本次预测集中共有36个类别0，57个类别1，7个类别2
    正确预测个数：94, 正确率：0.940000
    =========
    第18轮平均loss:0.468705
    本次预测集中共有35个类别0，50个类别1，15个类别2
    正确预测个数：88, 正确率：0.880000
    =========
    第19轮平均loss:0.456445
    本次预测集中共有31个类别0，53个类别1，16个类别2
    正确预测个数：91, 正确率：0.910000
    =========
    第20轮平均loss:0.445071
    本次预测集中共有34个类别0，51个类别1，15个类别2
    正确预测个数：89, 正确率：0.890000
    [[0.56, 1.0419725303649903], [0.58, 0.9389978256225586], [0.7, 0.8783432097434998], [0.67, 0.8266444535255432], [0.78, 0.7807808015346527], [0.72, 0.7398336427211761], [0.84, 0.7031631523370743], [0.87, 0.670227635383606], [0.79, 0.6405569955110549], [0.86, 0.6137440012693405], [0.8, 0.5894372901916504], [0.85, 0.5673345295190811], [0.91, 0.547175880432129], [0.85, 0.5287377995252609], [0.89, 0.5118274569511414], [0.84, 0.4962784694433212], [0.94, 0.4819463350772858], [0.88, 0.46870528686046603], [0.91, 0.4564454814195633], [0.89, 0.44507056015729907]]



    
![png](output_0_2.png)
    



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/vk/c6_mn1hd72x8nm30y52hjz540000gn/T/ipykernel_76137/3497016100.py in <module>
         92                 [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
         93                 [0.19349776,0.59416669,0.92579291,0.41567412,0.7358894]]
    ---> 94     predict("model.pt", test_vec)
    

    NameError: name 'predict' is not defined



```python

```
