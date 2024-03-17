import numpy as np
import torch.nn as nn
import torch
from units import padding
from transformers import BertModel
from config import config

class Edit_match():
    def __init__(self,match_pool):
        self.match_pool=match_pool
    def edit_distance(self,str1, str2):
        '''
        :param str1:
        :param str2:
        :return:
        '''
        string1 = str1
        string2 = str2
        matrix = np.zeros((len(string1) + 1, len(string2) + 1))
        for i in range(len(string1) + 1):
            matrix[i][0] = i
        for j in range(len(string2) + 1):
            matrix[0][j] = j
        for i in range(1, len(string1) + 1):
            for j in range(1, len(string2) + 1):
                if string1[i - 1] == string2[j - 1]:
                    d = 0
                else:
                    d = 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
        return matrix[len(string1)][len(string2)]

    def similarity_based_on_edit_distance(self,str1, str2):
        dis = self.edit_distance(str1, str2)
        sim = 1 - dis / max(len(str1), len(str2))
        return sim

    def query(self,user_question):
        match_answer=None
        best_match=-1

        for match_pair in self.match_pool:
            questions,target=match_pair[0],match_pair[1]
            scores=[self.similarity_based_on_edit_distance(user_question,question) for question in questions]
            score=max(scores)

            if score>best_match:
                best_match=score
                match_answer=target
        return [match_answer,best_match]

class Representation(nn.Module):
    '''
    表示型的文本匹配
    '''
    def __init__(self,vocab,config):
        super().__init__()

        self.vocab=vocab
        self.config=config

        # 词嵌入层
        self.embedding=nn.Embedding(len(vocab),self.config['embedding_dim'],padding_idx=0)
        #表示层
        self.lstm=nn.LSTM(input_size=self.config['embedding_dim'],hidden_size=self.config['hidden_dim'],batch_first=True)
        #输出层
        self.fc=nn.Linear(in_features=self.config['hidden_dim'],out_features=self.config['out_dim'])

        self.pool=nn.AvgPool1d(kernel_size=self.config['max_len'])

        self.dropout = nn.Dropout(0.5)

    def forward(self,x):

        #输入为batch个句子，返回encode后的batch个句子向量
        x=self.embedding(x)
       # x,_=self.lstm(x)
        #x=self.dropout(x)
        x=self.fc(x)
        #shape:[batch,out_dim]
        x=self.pool(x.transpose(1,2)).squeeze(-1)
        return x

class Match_layer(nn.Module):
    '''
    该类用来计算两个样本的相似度，采用余弦损失函数
    '''
    def __init__(self,vocab,config):
        super().__init__()
        self.vocab=vocab
        self.config=config
        self.embedding_representation=Representation(vocab,config)
        self.loss= nn.CosineEmbeddingLoss()

    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def triplet_loss(self,margin,p1,p2,n):
        '''
        :param margin:
        :param p1: 第一个正样本
        :param p2: 第二个正样本
        :param n: 第三个正样本
        :return:
        '''

        ap=self.cosine_distance(p1,p2)
        np=self.cosine_distance(p1,n)

        diff=ap-np+margin

        loss=torch.mean(diff[diff.gt(0)])

        return loss

    def forward(self,x1,x2=None,label=None):

        #计算损失函数
        if x2 is not None:
            x1=self.embedding_representation(x1)
            x2=self.embedding_representation(x2)
            if label is not None:
                #如果输入了标签，计算损失函数
                label=label.squeeze()
                return self.loss(x1,x2,label)

                #如果没有输入标签，计算基于余弦相似度的距离，然后计算triplet loss






            elif label is None:
                return self.cosine_distance(x1,x2)
        else:
            #返回表示层输出
            return self.embedding_representation(x1)

class Interaction_layer(nn.Module):
    def __init__(self,vocab,config):
        super().__init__()
        self.vocab=vocab
        self.config=config
        self.embedding=nn.Embedding(len(vocab),config['embedding_dim'])
        self.lstm=nn.LSTM(input_size=config['embedding_dim'],hidden_size=config['hidden_dim']*2,batch_first=True)
        self.pool=nn.MaxPool1d(config['max_len'])
        self.relu=nn.ReLU()
        self.fc=nn.Linear(in_features=config['hidden_dim']*2,out_features=config['hidden_dim'])
        self.classfier=nn.Linear(in_features=config['hidden_dim'],out_features=2)
        self.softmax=nn.Softmax(dim=1)
        self.loss=nn.CrossEntropyLoss()

    def forward(self,sentence,label=None):

        x=self.embedding(sentence)
        x,_=self.lstm(x)
        x=self.pool(x.transpose(1,2)).squeeze()
        x=self.relu(x)
        x=self.fc(x)
        x=self.classfier(x)

        if label is not None:
            label = label.squeeze()
            loss_value=self.loss(x,label)
            return loss_value
        else:

            return torch.softmax(x, dim=-1)[:, 1]

