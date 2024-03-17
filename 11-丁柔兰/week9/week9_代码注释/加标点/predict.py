# -*- coding: utf-8 -*-
# 导入PyTorch深度学习库、用于正则表达式的re库、用于处理JSON的json库，以及numpy数学库。
# defaultdict来自collections模块，用于创建带有默认值的字典。Config从config模块导入，TorchModel从model模块导入
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel

"""
模型效果测试=============================整个脚本提供了一种方式来测试一个训练好的命名实体识别模型，它可以对任意文本进行实体标注，并输出标注结果================================================
"""


class SentenceLabel:
    # 类的初始化方法，接收配置对象config和模型文件路径model_path作为参数
    def __init__(self, config, model_path):
        # 在初始化方法中，加载了模式信息、索引到标签的映射字典和词汇表
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = self.load_vocab(config["vocab_path"])
        # 创建一个TorchModel实例，加载训练好的模型状态，并设置模型为评估模式
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")  # 打印一条提示信息表示模型加载完成

    def load_schema(self, path):  # 定义了一个名为load_schema的方法，用于加载模式信息
        with open(path, encoding="utf8") as f:  # 以UTF-8编码格式打开模式文件，读取JSON数据，并设置配置中的类别数量
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema  # 返回加载的模式信息

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}  # 初始化一个空字典用于存储词汇表
        with open(vocab_path, encoding="utf8") as f:  # 以UTF-8编码格式打开词汇表文件
            for index, line in enumerate(f):  # 遍历文件中的每一行及其索引
                token = line.strip()  # 去除行首尾空白字符
                # 将词汇及其索引（从1开始计数）存入字典，0索引留给padding位置
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        # 设置配置中的词汇表大小
        self.config["vocab_size"] = len(token_dict)
        return token_dict  # 返回词汇表字典

    # 定义了一个名为predict的方法，用于预测输入句子的实体标签
    def predict(self, sentence):
        input_id = []  # 初始化一个空列表用于存储输入的词索引
        for char in sentence:  # 遍历句子中的每一个字符
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))  # 将字符转换为索引，如果词汇不在词汇表中，则使用未知字符"[UNK]"的索引
        with torch.no_grad():  # 在不追踪梯度的情况下执行模型的前向传播
            res = self.model(torch.LongTensor([input_id]))[0]  # 将输入数据转换为长整型张量，并进行前向传播，获取预测结果
            res = torch.argmax(res, dim=-1)  # 在最后一个维度上使用argmax获取预测的类别索引
        labeled_sentence = ""  # 初始化一个空字符串用于存储标注后的句子
        for char, label_index in zip(sentence, res):  # 遍历句子中的每个字符和对应的预测标签索引
            labeled_sentence += char + self.index_to_sign[int(label_index)]  # 将字符和对应的标签拼接起来，构成标注后的句子
        return labeled_sentence  # 返回标注后的句子


if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_10.pth")  # 创建SentenceLabel的实例，传入配置和模型文件路径

    # 给定一个句子，使用predict方法进行预测，并打印结果
    sentence = "客厅的颜色比较稳重但不沉重相反很好的表现了欧式的感觉给人高雅的味道"
    res = sl.predict(sentence)
    print(res)
    # 再次给定另一个句子，进行预测，并打印结果
    sentence = "双子座的健康运势也呈上升的趋势但下半月有所回落"
    res = sl.predict(sentence)
    print(res)
