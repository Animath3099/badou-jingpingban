# -*- coding: utf-8 -*-
# ===============这段代码的主要作用是定义了一个数据处理流程，它可以随机采样生成训练数据，并能够加载测试数据集。它是自定义的，特别适用于特定的文本处理任务===============

import json  # json用于处理JSON格式数据
import re  # re用于正则表达式
import os  # os用于操作系统接口
import torch  # torch是PyTorch深度学习框架
import random  # random用于生成随机数
import jieba  # jieba用于中文分词
import numpy as np  # numpy用于科学计算
from torch.utils.data import Dataset, DataLoader  # DataLoader用于数据加载
from collections import defaultdict  # defaultdict用于创建带有默认值的字典

"""
数据加载
"""


class DataGenerator:  # 定义了一个名为DataGenerator的类，用于数据生成
    def __init__(self, data_path, config):
        self.config = config  # 加载配置信息
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])  # 加载词汇表
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])  # 加载数据架构
        self.train_data_size = config["epoch_data_size"]  # 加载设置训练数据大小# 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  # 加载数据类型# 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()  # 加载调用self.load()方法加载数据

    # 定义了一个名为load的方法，用于加载数据
    def load(self):
        # 初始化存储数据的变量self.data和知识库的变量self.knwb
        self.data = []
        self.knwb = defaultdict(list)
        # 打开并读取文件，每行作为一个JSON对象处理。如果行是字典类型，则被认为是训练集的数据
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    # 设置数据类型为训练集，对问题进行编码并转换为PyTorch张量，然后添加到知识库对应标签的列表中
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集
                else:
                    # 如果行是列表类型，则被认为是测试集的数据。同样进行编码，并将问题和标签索引添加到self.data中
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    # 定义了一个名为encode_sentence的方法，用于对文本进行编码
    def encode_sentence(self, text):
        # 根据配置文件中指定的词表路径，使用jieba进行分词或按字符进行编码，并将词或字符转换为对应的索引。如果词典中不存在，则使用[UNK]代替
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        # 调用self.padding方法对序列进行填充或截断，并返回编码后的序列
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    # 定义了一个名为padding的方法，用于补齐或截断输入的序列，使其长度等于配置中的最大长度max_length
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    # 实现了__len__魔术方法，返回数据集的大小
    def __len__(self):
        # 如果数据类型是训练集，则调用self.random_train_sample()方法生成随机样本；如果是测试集，则返回指定索引的数据
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    # 实现了__getitem__魔术方法，用于获取特定索引的数据项
    def __getitem__(self, index):
        # 如果数据类型是训练集，则调用self.random_train_sample()方法生成随机样本；如果是测试集，则返回指定索引的数据
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 依照一定概率生成负样本或正样本
    # 负样本从随机两个不同的标准问题中各随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    # 定义了一个名为random_train_sample的方法，用于随机生成训练样本
    def random_train_sample(self):
        # 获取所有标准问题的索引列表
        standard_question_index = list(self.knwb.keys())
        # 随机正样本
        # 根据配置中的正样本比例来随机决定生成正样本还是负样本
        # if random.random() <= self.config["positive_sample_rate"]:
        # 如果决定生成正样本，则从同一类别中随机选取两个问题作为正样本对
        # p = random.choice(standard_question_index)
        p, n = random.sample(standard_question_index, 2)
        if len(self.knwb[p]) == 1:
            s1 = s2 = self.knwb[p][0]
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
        s3 = random.choice(self.knwb[n])
        # 如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        # if len(self.knwb[p]) < 3:
        #     return self.random_train_sample()
        # else:
        #     s1, s2, s3 = random.sample(self.knwb[p], 3)
        #     return [s1, s2, s3]
        # 随机负样本
        # else:  # 如果决定生成负样本，则分别从两个不同类别中随机选取问题作为负样本对
        #     p, n, a = random.sample(standard_question_index, 3)
        #     s1 = random.choice(self.knwb[p])
        #     s2 = random.choice(self.knwb[n])
        #     s3 = random.choice(self.knwb[a])
        return [s1, s2, s3]


# 加载字表或词表
# 定义了一个名为load_vocab的函数，用于加载词汇表
def load_vocab(vocab_path):
    # 读取词汇表文件，创建一个词到索引的字典。由于0被用作padding，所以索引从1开始
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 加载schema
# 定义了一个名为load_schema的函数，用于加载标签架构
def load_schema(schema_path):
    # 读取标签架构文件，并返回解析后的JSON对象
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
# 定义了一个名为load_data的函数，用于加载数据并返回一个DataLoader对象
def load_data(data_path, config, shuffle=True):
    # 实例化DataGenerator，然后使用DataLoader将其封装起来，设置批量大小和是否打乱数据
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    # 从config模块导入配置，创建DataGenerator实例，并打印索引为1的数据项
    from config import Config

    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
