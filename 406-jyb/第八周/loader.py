import os
import json
import numpy as np
import torch
import random
from transformers import BertTokenizer
from torch.utils.data import Dataset,DataLoader
from collections import defaultdict

class load_qa():
    def __init__(self):
        self.train_data_path=r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\train.json"
        self.valid_data_path=r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\valid.json"
        self.class_label_path=r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\schema.json"
        self.match_pool=[]
        self.valid_pool=[]
        self.target_list=[]
    def load_train(self):
        with open(self.train_data_path,encoding='utf-8') as f:
            for line in f:
                line=json.loads(line)
                if isinstance(line,dict):
                    questions=line['questions']
                    target=line['target']
                    self.target_list.append(target)
                    self.match_pool.append([questions, target])
        f.close()
        print("训练数据加载完毕，格式为：【 【问题集合，答案】。。。】")
        return
    def load_valid_data(self):
        with open(self.valid_data_path,encoding='utf-8') as f:
            for line in f:
                line=json.loads(line)
                if isinstance(line,list):
                    self.valid_pool.append(line)
        f.close()
        print("测试数据读取完毕，格式为：【【问题，答案】。。。】")

class R_loader():
    def __init__(self,config,vocab):
        self.config=config
        self.read_type=None
        self.schema_path=config['schema_path']
        self.schema=self.load_schema()
        self.vocab=vocab
        self.valid_data=[]
        self.class_questions = defaultdict(list)
    def load_data(self,path):
        #各类任务对应列表。{类别：【问题】}

        with open(path,encoding='utf-8') as f:
            for line in f:
                line=json.loads(line)
                if isinstance(line,dict):

                    #表示正在进行训练
                    self.read_type="train"

                    #加载目标和问题列表
                    questions=line['questions']
                    target=line['target']
                    for question in questions:
                        #句子向量化,padding

                        question=self.encode_sentence(question)
                        question=self.padding(question)
                        question=torch.LongTensor(question)
                        self.class_questions[self.schema[target]].append(question)
                elif isinstance(line,list):
                    self.read_type="valid"
                    question,target=line
                    question=self.encode_sentence(question)
                    question=self.padding(question)
                    question=torch.LongTensor(question)
                    self.valid_data.append([question,torch.LongTensor([self.schema[target]])])

    def __getitem__(self, item):
        if self.read_type=="train":
            #随机选取样本
            return self.random_choice_sample()
        else:
            return self.valid_data[item]


    def __len__(self):
        if self.read_type=="train":
            return self.config["epoch_data_size"]
        elif self.read_type=="valid":
            return len(self.valid_data)
    def load_schema(self):
        '''
        加载类别表
        :return:
        '''
        with open(self.schema_path, encoding="utf8") as f:
            return json.loads(f.read())

    def encode_sentence(self,sentence):
        '''
        将句子转化为向量
        :param sentence:
        :return:
        '''
        vector=[self.vocab.get(word,0) for word in sentence]
        return vector

    def padding(self,vector):
        '''
        将数值类型的句子补齐或者阶段
        :param vector:
        :return:
        '''

        max_len=self.config['max_len']

        if len(vector)>max_len:
            vector=vector[:max_len]
            return vector
        elif len(vector)<max_len:
            vector.extend([0]*(max_len-len(vector)))
            return vector
        else:
            return vector
    def random_choice_sample(self):
        '''
              流程：根据概率，决定选取正样本还是负样本

              正样本:随机选取一个类，从这个类中随机选取两个问题

              负样本，随机选取两个类，从每个类中随机选取一个问题
              :return:
        '''

        class_list=list(self.class_questions.keys())

        #小于这个，选取正样本
        if random.random()<self.config['positive_rate']:
            #决定选取哪个类
            c=random.choice(class_list)

            #如果该类下的问题不够两个，需要重新选取
            if len(self.class_questions[c])<2:
                return self.random_choice_sample()
            else:
                q1,q2=random.sample(self.class_questions[c],2)
                return [q1,q2,torch.LongTensor([1])]
        else:
            c1,c2=random.sample(class_list,2)

            q1=random.choice(self.class_questions[c1])
            q2=random.choice(self.class_questions[c2])

            return [q1,q2,torch.LongTensor([-1])]

class sim_loader():
    def __init__(self,config):
        self.config=config
        self.read_type=None
        self.class_questions=defaultdict(list)
        self.schema = load_schema(config["schema_path"])
        self.valid_data=[]
        self.tokenizer=BertTokenizer(config['vocab_path'])

    def load(self,path):
        if "train" in path:
            self.read_type='train'
            with open(path,'r',encoding='utf-8') as f:
                for line in f:
                    line=json.loads(line)
                    assert isinstance(line, dict)
                    label=line['target']
                    questions=line['questions']

                    for question in questions:
                        self.class_questions[self.schema[label]].append(question)
        else:
            self.read_type="valid"
            with open(path,'r',encoding='utf-8') as f:
                for line in f:
                    line=json.loads(line)
                    assert isinstance(line,list)
                    question,label=line
                    label=self.schema[label]
                    self.valid_data.append([question,torch.LongTensor([label])])

    def sentence_encoder(self,text1,text2):

        concat_text=self.tokenizer.encode(text1,text2,truncation='longest_first',
                                         max_length=self.config['max_len'],
                                         padding='max_length',)

        return concat_text

    def __len__(self):
        if self.read_type=="train":
            return self.config['epoch_data_size']
        else:
            return len(self.valid_data)

    def __getitem__(self, item):
        if self.read_type=="train":
            return self.random_sample()
        else:
            return self.valid_data[item]

    def random_sample(self):
        #选取正样本

        label_list=list(self.class_questions.keys())

        if random.random()<self.config['positive_rate']:
            c=random.choice(label_list)
            if len(self.class_questions[c])<2:
                return self.random_sample()
            else:
                p1,p2=random.sample(self.class_questions[c],2)
                concat_text=self.sentence_encoder(p1,p2)
                return [torch.LongTensor(concat_text),torch.LongTensor([1])]
        else:
            c1,c2=random.sample(label_list,2)

            n1=random.choice(self.class_questions[c1])
            n2=random.choice(self.class_questions[c2])

            concat_text=self.sentence_encoder(n1,n2)
            return [torch.LongTensor(concat_text),torch.LongTensor([0])]

class triple_loader():
    def __init__(self,config):
        self.config=config
        self.read_type=None#训练集还是测试集
        self.vocab=load_vocab()
        self.schema=load_schema(config['schema_path'])
        self.data=[]

    def load_data(self,path):
        self.class_questions = defaultdict(list)
        with open(path,'r',encoding='utf8') as f:
            for line in f:
                line=json.loads(line)
                if isinstance(line,dict):
                    self.read_type="train"
                    questions,target=line['questions'],line['target']
                    for question in questions:
                        #转换为数字类型的list
                        question=self.padding(self.sentence2vector(question))
                        label=self.schema[target]
                        #{标签:任务列表}
                        self.class_questions[label].append(question)
                elif isinstance(line,list):
                    self.read_type="valid"
                    question,target=line[0],line[1]
                    question=self.padding(self.sentence2vector(question))
                    self.data.append([torch.LongTensor([question]),torch.LongTensor([target])])

    def __len__(self):
        if self.read_type=="train":
            return self.config['epoch_data_size']
        else:
            return len(self.data)
    def __getitem__(self, item):
        if self.read_type=="train":
            return self.random_sample()
        else:
            return self.data[item]
    def random_sample(self):
        label_list=list(self.class_questions.keys())

        p,n=random.sample(label_list,2)

        if len(self.class_questions[p])<2:
            return self.random_sample()

        ps1,ps2=random.sample(self.class_questions[p],2)

        ns=random.choice(self.class_questions[n])

        return [torch.LongTensor(ps1),torch.LongTensor(ps2),torch.LongTensor(ns)]

    def padding(self,sentence):
        if len(sentence)<self.config['max_len']:
            sentence.extend([0]*(self.config['max_len']-len(sentence)))
        elif len(sentence)>self.config['max_len']:
            sentence=sentence[:self.config['max_len']]
        return sentence

    def sentence2vector(self,question):
        vector=[self.vocab.get(word,0) for word in question]
        return vector

def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

def load_vocab(vocab_path=r"E:\python\bert_file\vocab.txt"):
    '''
    这里采用bert中文版的字典
    :param vocab_path:
    :return:
    '''
    vocab_dict={}
    with open(vocab_path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            vocab_dict[line]=len(vocab_dict)
    return vocab_dict
