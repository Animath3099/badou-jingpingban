# -*- coding: utf-8 -*-


# 导入了PyTorch库以及神经网络模块（torch.nn），优化器（Adam和SGD），以及用于处理变长序列的函数
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
建立网络模型结构
"""


# 定义了一个名为SentenceEncoder的类，它继承了nn.Module。
# 这个类将被用作编码单个句子的组件
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]  # 从配置字典config中获取相关参数：hidden_size定义了网络隐藏层的大小
        vocab_size = config["vocab_size"] + 1  # vocab_size定义了词汇表的大小（并加1考虑到padding token）
        max_length = config["max_length"]  # max_length定义了输入句子的最大长度
        # 创建一个词嵌入层，用于将单词索引转换为词向量。
        # padding_idx=0表示索引为0的单词被用作padding token，它的向量表示将会是零向量
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)#LSTM来处理序列数据
        # 创建了一个线性层self.layer，其输入和输出维度均为hidden_size
        self.layer = nn.Linear(hidden_size, hidden_size)
        # dropout层，用于在训练期间随机丢弃一半的特征，以减少过拟合
        self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)  # 通过词嵌入层将输入的单词索引转换为向量
        # 使用LSTM处理
        # x, _ = self.lstm(x)
        # 使用线性层:通过一个线性层
        x = self.layer(x)
        # 对线性层的输出进行最大池化操作，然后移除多余的维度
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x  # 返回池化后的结果


# 定义了一个名为SiameseNetwork的类，
# 它也继承了nn.Module,这个类将被用作整个孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        # 在孪生网络中，创建了一个SentenceEncoder实例来编码句子，
        # 并定义了一个余弦嵌入损失函数，用于计算两个句子编码之间的相似度损失
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        # 定义cosine_distance方法，它首先对输入的两个张量进行归一化处理
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        # 计算归一化张量的点积得到余弦相似度，然后返回1减去余弦相似度得到余弦距离
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    # 定义了一个用于计算三元组损失的方法，
    # 这在训练时用于区分anchor（a）、positive（p）和negative（n）样本
    def cosine_triplet_loss(self, a, p, n, margin=None):
        # 计算anchor与positive和negative样本之间的余弦距离
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        # 计算正样本和负样本之间的距离差值，并加上一个边界值（margin）。
        # 仅对大于0的diff值求平均，这是triplet loss的典型实现
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])  # greater than

    # sentence : (batch_size, max_length)
    # 再次定义forward方法，用于处理输入的句子并可能计算损失
    def forward(self, sentence1, sentence2=None, sentence3=None, target=None):
        # 同时传入两个句子
        if sentence2 is not None and sentence3 is not None:  # 如果提供了两个句子，分别计算它们的编码
            vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            vector3 = self.sentence_encoder(sentence3)
            return self.cosine_triplet_loss(vector1, vector2, vector3)
            # 如果有标签，则计算loss
            # 如果提供了目标标签，计算loss并返回损失值。如果没有标签，返回余弦距离
            # if target is not None:
            #     return self.cosine_triplet_loss(vector1, vector2, vector3)
            # # 如果无标签，计算余弦距离
            # else:
            #     return self.cosine_distance(vector1, vector2)
        # 单独传入一个句子时，认为正在使用向量化能力
        # 如果只有一个句子，返回该句子的编码
        else:
            return self.sentence_encoder(sentence1)


# 定义一个函数choose_optimizer，用于根据配置选择优化器
def choose_optimizer(config, model):
    # 从配置中获取优化器的类型和学习率
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    # 根据配置创建并返回相应的优化器实例
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    # 导入一个配置文件
    from config import Config

    # 并设置词汇表大小和最大句子长度
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    # 使用配置信息实例化一个孪生网络模型
    model = SiameseNetwork(Config)
    # 创建两个示例句子和它们的标签（可能表示句子是否相似）
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    # 将句子通过模型前向传播，计算损失，并打印输出
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())# 打印模型的状态字典，即模型的参数
