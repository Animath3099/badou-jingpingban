#coding:utf8
import torch 
import torch.nn as nn
import numpy as np 
import random
import matplotlib.pyplot as plt
import time
# class TorchModel(nn.Module):
#     #这里构造全连接网络，每隔两层网络增加归一化层，
#     def __init__(self, input_size, hidden_size, output_size,drop_out_rate):
#         super(TorchModel, self).__init__()
#         self.layer1 = nn.Linear(input_size,2*hidden_size)
#         self.layer2 = nn.Linear(2*hidden_size,3*output_size)
#         self.Bn_1 = nn.BatchNorm1d(3*output_size)
#         self.layer3 = nn.Linear(3*output_size,2*output_size)
#         self.layer4 = nn.Linear(2*output_size,2*hidden_size)
#         self.dropout1 = nn.Dropout(drop_out_rate)
#         self.Bn_2 = nn.BatchNorm1d(2*hidden_size)
#         self.layer5 = nn.Linear(2*hidden_size,output_size)
#         # self.Bn_3 = nn.BatchNorm1d(output_size)
#         self.layer6 = nn.Linear(output_size,4*hidden_size)
#         self.Bn_3 = nn.BatchNorm1d(4*hidden_size)
#         self.layer7 = nn.Linear(4*hidden_size,3*output_size)
#         self.layer8 = nn.Linear(3*output_size,output_size)
#         self.dropout2 = nn.Dropout(drop_out_rate)
#         self.Bn_4 = nn.BatchNorm1d(output_size)
#         self.layer11 = nn.Linear(output_size,2*hidden_size)
#         self.layer21 = nn.Linear(2*hidden_size,3*output_size)
#         self.Bn_11 = nn.BatchNorm1d(3*output_size)
#         self.layer31 = nn.Linear(3*output_size,2*output_size)
#         self.layer41 = nn.Linear(2*output_size,2*hidden_size)
#         self.dropout11 = nn.Dropout(drop_out_rate)
#         self.Bn_21 = nn.BatchNorm1d(2*hidden_size)
#         self.layer51 = nn.Linear(2*hidden_size,output_size)
#         # self.Bn_31 = nn.BatchNorm1d(output_size)
#         self.layer61 = nn.Linear(output_size,4*hidden_size)
#         self.Bn_31 = nn.BatchNorm1d(4*hidden_size)
#         self.layer71 = nn.Linear(4*hidden_size,3*output_size)
#         self.layer81 = nn.Linear(3*output_size,output_size)
#         self.dropout21 = nn.Dropout(drop_out_rate)
#         self.Bn_41 = nn.BatchNorm1d(output_size)
#         self.activation = torch.relu
#         self.loss = nn.functional.cross_entropy

#     def forward(self, x, y=None):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.Bn_1(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = torch.relu(x)
#         x = self.dropout1(x)
#         x = self.Bn_2(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.Bn_3(x)
#         x = self.layer7(x)
#         x = self.layer8(x)
#         x = torch.relu(x)
#         x = self.dropout2(x)
#         x = self.Bn_4(x)
#         x = self.layer11(x)
#         x = self.layer21(x)
#         x = self.Bn_11(x)
#         x = self.layer31(x)
#         x = self.layer41(x)
#         x = torch.relu(x)
#         x = self.dropout11(x)
#         x = self.Bn_21(x)
#         x = self.layer51(x)
#         x = self.layer61(x)
#         x = self.Bn_31(x)
#         x = self.layer71(x)
#         x = self.layer81(x)
#         x = torch.relu(x)
#         x = self.dropout21(x)

#         y_pred = x
#         # x = self.Bn_41(x)
#         # y_pred = self.activation(x)
#         y_pred, indices = torch.max(y_pred,axis=1)
#         if y is None:
#             return indices
#         else:
#             return self.loss(y_pred,y)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training.")
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out_rate):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 2*hidden_size)
        self.Bn_1 = nn.BatchNorm1d(2*hidden_size)
        self.dropout1 = nn.Dropout(drop_out_rate)

        self.layer2 = nn.Linear(2*hidden_size, 3*output_size)
        self.Bn_2 = nn.BatchNorm1d(3*output_size)
        self.dropout2 = nn.Dropout(drop_out_rate)

        self.layer3 = nn.Linear(3*output_size, 2*output_size)
        self.Bn_3 = nn.BatchNorm1d(2*output_size)
        self.dropout3 = nn.Dropout(drop_out_rate)

        self.layer4 = nn.Linear(2*output_size, 2*hidden_size)
        self.Bn_4 = nn.BatchNorm1d(2*hidden_size)
        self.dropout4 = nn.Dropout(drop_out_rate)

        self.layer5 = nn.Linear(2*hidden_size, output_size)
        self.Bn_5 = nn.BatchNorm1d(output_size)
        self.dropout5 = nn.Dropout(drop_out_rate)

        self.activation = torch.relu
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.activation(self.Bn_1(self.layer1(x)))
        x = self.dropout1(x)
        
        x = self.activation(self.Bn_2(self.layer2(x)))
        x = self.dropout2(x)

        x = self.activation(self.Bn_3(self.layer3(x)))
        x = self.dropout3(x)

        x = self.activation(self.Bn_4(self.layer4(x)))
        x = self.dropout4(x)

        x = self.activation(self.Bn_5(self.layer5(x)))
        x = self.dropout5(x)

        if y is not None:
            return self.loss(x, y)
        else:
            _, indices = torch.max(x, axis=1)
            return indices

#数据标签生成，这里的生成规则是label是最大值索引
def buid_sample(num):
    """
        num：生成随机数个数
        lael为对应最大值所在的索引
    """
    input_data = np.random.random(num)
    target_data = np.argsort(input_data)[::-1]
    return input_data, target_data[0]
        
        
#生成训练数据，这里生成数据返回为torch张量
def build_dataset(num_of_dataset, numrange):
    input_data, label = [], []
    for i in range(num_of_dataset):
        tmp_input, tmp_label = buid_sample(numrange)
        input_data.append(tmp_input)
        label.append(tmp_label)
    return torch.FloatTensor(input_data), torch.FloatTensor(label)

def evaluate(model,num_of_testdata,numofrange):
    model.eval()
    test_data, test_label = build_dataset(num_of_dataset=num_of_testdata,numrange=numofrange)
    correct, wrong = 0, 0 
    test_data, test_label = test_data.to(device), test_label.to(device)
    with torch.no_grad():
        y_pred = model(test_data)
        for y_p, y in zip(y_pred,test_label):
            if y == y_p:
                correct+=1
            else:
                wrong+=1
        return correct / np.add(correct,wrong)
    
    
def train():
    start_time = time.time()
    #参数初始化，设置epoch较小时效果不明显
    epoch = 100000
    num_of_dataset = 100000
    batch_size = 2000
    numrange = 10
    input_size = 10  
    hidden_size = 20
    output_size = 10
    lr = 1e-4
    drop_out_rate = 0.2
    log = []
    #实例化模型和优化器，并且准备训练数据
    model = TorchModel(input_size=input_size,hidden_size=hidden_size,output_size=output_size,drop_out_rate=drop_out_rate)
    model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=1e-5)
    train_data, train_label = build_dataset(num_of_dataset=num_of_dataset,numrange=numrange)
    best_acc = 0
    flag = 0
    for i in range(epoch):
        model.train()
        watch_loss = []
        for batch_index in range(num_of_dataset // batch_size): 
            tmp_train_data, tmp_train_label = train_data[batch_size*batch_index:batch_size*(batch_index+1)], train_label[batch_index*batch_size:batch_size*(batch_index+1)].long()
            tmp_train_data, tmp_train_label =tmp_train_data.to(device),tmp_train_label.to(device)
            loss = model(tmp_train_data,tmp_train_label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("\n第%d轮的平均loss:%f"%(i+1,np.mean(watch_loss)))
        acc = evaluate(model,100, 10)
        log.append([acc,float(np.mean(watch_loss))])
        print("acc: %f"%acc)
        if best_acc < acc:
            best_acc = acc
            best_loss = np.mean(watch_loss)
            tmp_state_dict = model.state_dict()
            flag = 0
        elif flag > 1000:
            torch.save(tmp_state_dict,"model.pt")
            print("best_acc:%f,best_loss:%f" % (best_acc, best_loss))
            break
        else:
            flag += 1 
        # log.append([acc,float(np.mean(watch_loss))])
        # print("acc: %f"%acc)
    torch.save(model.state_dict(),'model.pt')
    during_time = time.time() - start_time
    # print(log)
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')
    plt.legend()
    plt.show()
    print("during_time:%f",during_time)
    return 

def predict(model_path, input_data,input_label):
    input_size = 10
    hidden_size = 20
    output_size = 10
    drop_out_rate = 0.2
    model = TorchModel(input_size=input_size,hidden_size=hidden_size,output_size=output_size,drop_out_rate=drop_out_rate)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    # print(model.state_dict())
    
    model.eval()

    with torch.no_grad():
        correct_pred = 0
        input_data_tensor = torch.FloatTensor(input_data).to(device)
        result = model.forward(input_data_tensor)
        # correct_pred = 0 
        # input_data = input_data.to(device)
        # result = model.forward(torch.FloatTensor(input_data))
        for data, res, label in zip(input_data,result,input_label): 
            if res == label:
                correct_pred += 1
            print("input_data: %s, 预测类别：%d,实际类别：%f" % (data, res, label))
        # 这里acc的计算只是为了看一下模型的泛化程度，不过可能存在测试数据在训练数据中，可以同批次生成，按比例划分测试数据和训练数据
        print("acc:%f"%(correct_pred / result.shape[0]))


if __name__ == "__main__":
    # train()
    test_data,test_label = build_dataset(100,10)
    predict("model.pt",test_data,test_label)


'''
best_acc:0.910000,best_loss:1.662065
'''