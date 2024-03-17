import torch
import pandas as pd
import numpy as np
import jieba
from torch.optim import Adam
import re
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict
from Model import Fast_text,LSTM,CNN

def remove_NoneSence(sentences):
    '''
    :param senteces:是一个列表，里面存储的元素是字符串
    :return: 去除列表中每个字符串中的标点符号，需要去除的标点符号在remove_chars中
    '''
    remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    new_sentences=[]
    for sentence in sentences:
        #如果字符串中包含标点符号，则将标点符号去除
        new_sentences.append(re.sub(remove_chars,'',sentence))
    return new_sentences
def build_vocal(context):
    '''
    :param context:pd.dataframe中的一列
    :return:
    '''

    vocab=defaultdict(int)

    #去除停用词
    context=remove_NoneSence(context)
    count=1
    for sentence in context:
        for word in sentence:
            if word not in vocab.keys():
                vocab[word]=count
                count+=1
    vocab['padding']=0
    return vocab
def sentence2vector(sentences,vocal,max_length):
    vectors=[]
    for sentence in sentences:
        vector=[]
        for i in range(len(sentence)):
            if i >=max_length:
                break
            word=sentence[i]
            vector.append(vocal[word])
        if len(vector)<max_length:
            padding=[0]*(max_length-len(vector))
            vector.extend(padding)
        vectors.append(vector)
    return vectors
def split_data(X,Y):

    x_train,x_text,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=20)

    y_train,y_test=torch.tensor(y_train.values,dtype=torch.float),torch.tensor(y_test.values,dtype=torch.float)

    return x_train,x_text,y_train,y_test
def evaluate(model,x_test,y_test):
    model.eval()
    right=0
    wrong=0
    predictions=[]
    with torch.no_grad():
        pred=model(x_test)
        pred=torch.reshape(pred,(-1,2))
        pred=torch.argmax(pred,dim=1)
    for p,t in zip(pred,y_test):
        if p==t:
            right+=1
        else:
            wrong+=1
    accuracy=right/(right+wrong)
    print("ACC:",accuracy)
    return accuracy

if __name__ == '__main__':


    #构建字典
    data_path=r"E:\badouFile\第七周\week7 文本分类问题\week7 文本分类问题\文本分类练习数据集\文本分类练习.csv"
    data=pd.read_csv(data_path)
    context=data['review']
    vocab=build_vocal(context)


    #加载数据集
    x_train,x_test,y_train,y_test=split_data(X=data['review'],Y=data['label'])

    #定义参数
    word_dim=20
    #句子长度
    max_length=30
    batch_size=16
    epoch=20
    vocab_size=len(vocab.keys())
    batch_num=len(x_train)//batch_size

    #数据转换为向量
    x_train=sentence2vector(x_train,vocab,max_length)
    x_test=sentence2vector(x_test,vocab,max_length)
    x_train=torch.tensor(x_train)
    x_test=torch.tensor(x_test)


    model=CNN(vocab_size,word_dim,max_length)
    optim=Adam(model.parameters(),lr=0.001)

    # watch_loss=[]
    # for e in range(epoch):
    #
    #     for iter_batch in range(batch_num):
    #         x_batch=x_train[iter_batch*batch_size:(iter_batch+1)*batch_size]
    #         y_batch=y_train[iter_batch*batch_size:(iter_batch+1)*batch_size]
    #         optim.zero_grad()
    #
    #         output=model.forward(x_batch)
    #         loss=model.computeLoss(output,y_batch)
    #
    #         loss.backward()
    #         optim.step()
    #
    #         print("Epoch: {}, Iteration: {}, Loss: {}".format(e,iter_batch,loss.item()))
    #
    #         watch_loss.append(loss.item())
    #
    # torch.save(model,'cnn.pth')
    # plt.plot(watch_loss)
    # plt.show()

    model_choice=['cnn.pth','lstm.pth','lstm.pth']

    for model_path in model_choice:
        model=torch.load(model_path)
        print(model_path)
        evaluate(model,x_test,y_test)