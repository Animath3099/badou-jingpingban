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


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.sentences = []
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = [8] # bert cls_token
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                merge_sentenece="".join(sentenece)
                self.sentences.append(merge_sentenece)
                # input_ids = self.encode_sentence(sentenece)
                input_ids = self.tokenizer.encode(merge_sentenece, max_length=self.config["max_length"], padding="max_length",truncation=True)
                labels = self.padding(labels, -1)

                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    def decode_pattern(self,key,num,results,labels,sentence):
        for l in re.finditer(f"({num}+)",labels):
            s,e=l.span()
            results[key].append(sentence[s:e])

    def decode(self,sentence,labels):
        sentence="$"+sentence
        labels="".join([str(label) for label in labels[:len(sentence)+2]])
        results= defaultdict(list)
        self.decode_pattern('LOCATION',"04",results,labels,sentence)
        self.decode_pattern('ORGANIZATION',"15",results,labels,sentence)
        self.decode_pattern('PERSON',"26",results,labels,sentence)
        self.decode_pattern('TIME',"37",results,labels,sentence)
        return results
        
    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

