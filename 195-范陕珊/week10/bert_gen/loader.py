# -*- coding: utf-8 -*-

import json
import re
import os
from collections import defaultdict

import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""




#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus



class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)
        self.corpus = load_corpus(config.corpus)
        self.load()

    def build_sample(self, window_size):
        corpus=self.corpus
        start = random.randint(0, len(corpus) - 1 - window_size)
        end = start + window_size
        window = corpus[start:end]
        target = corpus[start+1:end + 1]  #输入输出错开一位
        # print(window, target)
        max_length=self.config.max_length
        x=self.tokenizer.encode("".join(window), max_length=max_length, padding="max_length",truncation=True)
        y=self.tokenizer.encode("".join(target), max_length=max_length, padding="max_length",truncation=True)
        return torch.LongTensor(x), torch.LongTensor(y)

    def build_dataset(self):
        dataset_x = []
        dataset_y = []
        for i in range(self.config.batch_size):
            x, y = self.build_sample( self.config.window_size)
            dataset_x.append(x)
            dataset_y.append(y)
        return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

    def load(self):
        for batch in range(int(self.config.train_sample / self.config.batch_size)):
            x, y = self.build_sample( self.config.window_size)  # 构建一组训练样本
            self.data.append([x,y])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



#用torch自带的DataLoader类封装数据
def load_data( config, shuffle=True):
    dg = DataGenerator( config)
    dl = DataLoader(dg.data, batch_size=config.batch_size, shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    data=load_data(Config)
    dg = DataGenerator( Config)

    print(data, len(data),len(dg.data))

