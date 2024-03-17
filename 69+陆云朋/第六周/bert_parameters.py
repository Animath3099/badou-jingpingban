import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel


class model_parameters:
    def __init__(self):
        self.n = 2  # 输入最大句子个数
        self.vocab = 21128  # 词表数目
        self.max_sequence_length = 512  # 最大句子长度
        self.embedding_size = 768  # embedding维度
        self.hide_size = 3072  # 隐藏层维数
        self.num_attention_heads = 12
        self.model = BertModel.from_pretrained(
            r"F:\Learn\NLP\bert-base-chinese", return_dict=False
        )

    def get_embedding_parameters_count(self):
        # 词表参数
        token_embedding_parameters = self.vocab * self.embedding_size
        # 句子参数
        segement_embedding_parameters = self.n * self.embedding_size
        # 位置参数
        position_embedding_parameters = self.max_sequence_length * self.embedding_size
        # layer_norm层参数
        layer_norm_parameters = self.embedding_size * 2
        return (
            token_embedding_parameters
            + segement_embedding_parameters
            + position_embedding_parameters
            + layer_norm_parameters
        )

    def get_attention_parameters_count(self):
        # w参数个数
        w_parameters = self.embedding_size * self.embedding_size
        # bias参数个数
        b_parameters = self.embedding_size
        # KQV三个参数总个数
        return (w_parameters + b_parameters) * 3

    def get_attention_out_parameters_count(self):
        # 线性曾输出参数
        line_layer_out_parameters = (
            self.embedding_size * self.embedding_size + self.embedding_size
        )
        # layer_norm层参数
        layer_norm_parameters = self.embedding_size * 2
        return line_layer_out_parameters + layer_norm_parameters

    def get_feed_forward_parameters_count(self):
        # 第一层线性层参数
        first_layer_parameters = (
            self.embedding_size * self.hide_size + self.hide_size
        )
        # 第二层线性层参数
        second_layer_parameters = (
            self.embedding_size * self.hide_size + self.embedding_size
        )
        # layer_norm层参数
        layer_norm_parameters = self.embedding_size * 2
        return first_layer_parameters + second_layer_parameters + layer_norm_parameters

    def get_pooler_parameters_count(self):
        return self.embedding_size * self.embedding_size + self.embedding_size

    def get_all_parameters_count(self):
        return (
            self.get_embedding_parameters_count()
            + self.get_attention_parameters_count()
            + self.get_attention_out_parameters_count()
            + self.get_feed_forward_parameters_count()
            + self.get_pooler_parameters_count()
        )

    def get_all_parameters_count_model(self):
        return sum(p.numel() for p in self.model.parameters())


def main():
    model = model_parameters()
    print("embedding层参数总个数为%d" % model.get_embedding_parameters_count())
    print("attention层参数总个数为%d" % model.get_attention_parameters_count())
    print("attention输出层参数总个数为%d" % model.get_attention_out_parameters_count())
    print("feed_forward层参数总个数为%d" % model.get_feed_forward_parameters_count())
    print("pooler层参数总个数为%d" % model.get_pooler_parameters_count())
    print("手动计算模型参数总个数为%d" % model.get_all_parameters_count())
    print("模型计算实际参数总个数为%d" % model.get_all_parameters_count_model())


if __name__ == "__main__":
    main()
