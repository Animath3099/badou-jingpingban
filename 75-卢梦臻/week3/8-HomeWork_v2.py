# _*_ coding: UTF-8 _*_
# @Time : 2024/2/10 22:53
# @Author : Yujez
# @File : 8-HomeWork_v2
# @Project : intro_to_ml

"""
输入一个字符串，根据字符a所在位置进行6分类
对比rnn和pooling做法

额外：考虑同个字符多次出现的情况
代办1：单条记录有无更好的写法
代办2：评价模型是否有更好的写法

"""
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as op
import matplotlib.pyplot as plt


# 0.创建字符集
def make_vocab():
    vocab={'pad':0}
    chars='asdfghjk'
    for i,char in enumerate(chars):
        vocab[char]=i+1
    vocab['unk']=len(vocab)
    return vocab
# 1.创建样本集
def build_single(vocab,str_length):
    x=random.sample(list(vocab.keys()),str_length)
    if 'a' in x:
        y=x.index('a')
    else:
        y=str_length
    x=[vocab.get(x1,'unk') for x1 in x]
    return x,y

def build_samples(num_samples,vocab,str_length):
    X,Y=[],[]
    for num in range(num_samples):
        x,y=build_single(vocab,str_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X),torch.LongTensor(Y)

# 3.建立网络层
class TorchModel(nn.Module):
    def __init__(self,vocab,str_length,char_dims):
        super(TorchModel,self).__init__()
        self.embd=nn.Embedding(len(vocab),char_dims)
        # 方式一：平均池化
        # self.pool=nn.AvgPool1d(str_length)
        # 方式二：rnn层
        self.rnn=nn.RNN(input_size=char_dims,hidden_size=char_dims,num_layers=2,batch_first=True)
        self.classify=nn.Linear(char_dims,str_length+1)
        self.loss=nn.functional.cross_entropy
    def forward(self,x,y=None):
        x=self.embd(x)          # batch_size*str_length*char_dims
        # 池化的处理方式
        # x=x.transpose(1,2)      # batch_size*char_dims*str_length
        # x=self.pool(x)          # batch_size*char_dims*1
        # x=x.squeeze()           # batch_size*char_dims
        # RNN的处理方式
        all_layer,ht=self.rnn(x)    # batch_size**char_dims
        x=all_layer[:,-1,:]
        # x=ht[-1,:,:]
        y_pred=self.classify(x)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred,y)

# 6.评价模型
def evaluate(model,vocab,str_length):
    model.eval()
    eval_nums=200
    x_eval,y_eval=build_samples(eval_nums,vocab,str_length)
    print(f"此次评价总样本数：{eval_nums}")
    y_pred=model.forward(x_eval)
    correct,wrong=0,0
    for y_p,y_e in zip(y_pred,y_eval):
        if torch.argmax(y_p)==y_e:
            correct+=1
        else:
            wrong+=1
    print(f"此轮正样本数：{correct}，准确率：{correct/(correct+wrong)}")
    print('------------------------------')
    return correct/(correct+wrong)

# 5.解耦模型实例化
def make_model(vocab,str_length,char_dims):
    return TorchModel(vocab,str_length,char_dims)

# 4.训练
def calculation():
    epoch_nums=20
    batch_size=50
    num_samples=1000
    lr=1e-3
    str_length=6
    char_dims=20
    vocab=make_vocab()
    model=make_model(vocab,str_length,char_dims)
    optim=op.Adam(model.parameters(),lr)
    log=[]
    for num in range(epoch_nums):
        model.train()
        loss_log=[]
        for batch_index in range(num_samples//batch_size):
            x,y=build_samples(batch_size,vocab,str_length)
            optim.zero_grad()
            loss=model.forward(x,y)
            loss.backward()
            optim.step()
            loss_log.append(loss.item())
        print(f"当前{num+1}轮，平均loss值：{np.mean(loss_log)}")
        acc=evaluate(model,vocab,str_length)
        log.append([acc,np.mean(loss_log)])
    draw(log)
    torch.save(model.state_dict(),'./support/homework_v2.pth')  # 保存参数
    with open('./support/vocab_hm_ver.json','w',encoding='utf8') as f:
        f.write(json.dumps(vocab,ensure_ascii=False, indent=2))

# 7.绘图
def draw(log):
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')
    plt.legend()
    plt.show()

# 8.预测
def predict(model_path,vocab_path,input_x):
    # 1.读取词表，还原词表
    with open(vocab_path,'r',encoding='utf8') as f:
        vocab=json.load(f)
    # vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    # 2.读取参数，还原成模型
    model=make_model(vocab,str_length=6,char_dims=20)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    temp_input=[]
    for chars in input_x:
        temp_input.append([vocab.get(char,vocab['unk']) for char in chars])
    with torch.no_grad():
        result=model.forward(torch.LongTensor(temp_input))
    for index,res in enumerate(result):
        print(f"输入：{input_x[index]}，预测a的位置：{np.argmax(res)}，实际预测值：{res}")


if __name__ == '__main__':
    calculation()
    test_strings = ["kijabc", "gijkbc", "gkijad", "kijhde"]
    predict('./support/homework_v2.pth','./support/vocab_hm_ver.json',test_strings)
