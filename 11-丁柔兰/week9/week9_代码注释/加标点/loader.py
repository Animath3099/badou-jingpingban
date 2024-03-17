# -*- coding: utf-8 -*-

import json  # 处理JSON的json库
import re  # 正则表达式的re库
import os  # 操作系统接口的os库
import torch  # PyTorch深度学习框架的torch
import random
import jieba  # 文本分词的jieba库
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class DataGenerator:
    # 类的初始化方法。当创建一个DataGenerator的实例时，这个方法会被调用。它接受数据文件的路径data_path和配置信息config作为参数
    def __init__(self, data_path, config):
        # 在初始化方法中，加载词汇表、模式信息，并设置了类的一些属性。load()方法将被调用以加载数据
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)
        self.max_length = config["max_length"]
        self.load()

    # 定义了load方法，用于从文件中加载并处理数据
    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if len(line) > self.max_length:  # 如果行的长度超过了最大长度配置
                    for i in range(len(line) // self.max_length):  # 将这一行分割成多个最大长度的段落
                        # 处理每段文本，并将其转换成id和标签的形式，然后将其作为长整型张量添加到数据列表中
                        input_id, label = self.process_sentence(line[i * self.max_length:(i + 1) * self.max_length])
                        self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
                else:
                    # 处理整行文本，并将其转换成id和标签的形式，然后将其作为长整型张量添加到数据列表中
                    input_id, label = self.process_sentence(line)
                    self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
        return

    # 定义了一个名为process_sentence的方法，用于处理每一行文本
    def process_sentence(self, line):
        # 初始化两个列表，一个用于存储去除标点的句子，另一个用于存储标签
        sentence_without_sign = []
        label = []
        # 遍历除了最后一个字符之外的所有字符
        for index, char in enumerate(line[:-1]):
            # 如果当前字符是一个标点符号，则跳过
            if char in self.schema:  # 准备加的标点，在训练数据中不应该存在
                continue
            # 将非标点的字符添加到去除标点的句子列表中
            sentence_without_sign.append(char)
            # 检查下一个字符是否是标点，如果是，将其对应的标签添加到标签列表中，否则添加0
            next_char = line[index + 1]
            if next_char in self.schema:  # 下一个字符是标点，计入对应label
                label.append(self.schema[next_char])
            else:
                label.append(0)
        # 断言去除标点的句子和标签列表的长度应该相等
        assert len(sentence_without_sign) == len(label)
        # 对句子进行编码并填充标签列表
        encode_sentence = self.encode_sentence(sentence_without_sign)
        label = self.padding(label, -1)
        # 再次断言编码后的句子和填充后的标签列表长度相等
        assert len(encode_sentence) == len(label)
        # 将去除标点的句子添加到句子列表
        self.sentences.append("".join(sentence_without_sign))
        # 返回编码后的句子和标签
        return encode_sentence, label

    # 定义了一个名为encode_sentence的方法，用于将文本编码为id
    def encode_sentence(self, text, padding=True):
        input_id = []  # 初始化一个空列表用于存储id
        if self.config["vocab_path"] == "words.txt":  # 如果使用词汇表进行编码
            for word in jieba.cut(text):  # 使用jieba对文本进行分词
                # 将每个词映射到它的id，如果词不在词汇表中，使用"[UNK]"的id
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:  # 如果不是使用词汇表，而是按字符编码
            for char in text:
                # 将每个字符映射到它的id，如果字符不在词汇表中，使用"[UNK]"的id
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:  # 如果需要，对id列表进行填充
            input_id = self.padding(input_id)
        return input_id  # 返回编码后的id列表

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        # 截断序列，使其长度不超过最大长度
        input_id = input_id[:self.config["max_length"]]
        # 如果序列长度不足，使用pad_token进行填充
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id  # 返回填充后的序列

    def __len__(self):
        return len(self.data)  # 返回数据列表的长度

    def __getitem__(self, index):
        return self.data[index]  # 定义了特殊方法__getitem__，它允许通过索引访问元素

    def load_schema(self, path):  # 定义了一个名为load_schema的方法，用于加载模式信息
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 加载字表或词表:定义了一个名为load_vocab的函数，用于加载词汇表
def load_vocab(vocab_path):
    token_dict = {}  # 初始化一个空字典用于存储词汇表
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):  # 遍历文件，获取每一行及其索引
            token = line.strip()  # 移除行首尾的空白字符
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始.将词汇及其索引（从1开始）添加到字典中
    return token_dict  # 返回词汇表字典


# 用torch自带的DataLoader类封装数据
# 定义了一个名为load_data的函数，它使用DataLoader来封装数据，以便于批处理和随机洗牌
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)  # 创建DataGenerator的实例
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)  # 使用DataLoader来加载DataGenerator实例，指定批大小和是否随机洗牌
    return dl  # 返回封装好的数据加载器


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../ner_data/train.txt", Config)  # 创建一个DataGenerator实例，用于加载和处理训练数据
