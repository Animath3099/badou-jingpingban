# -*- coding: utf-8 -*-
#导入PyTorch的核心库、神经网络模块、优化器模块中的Adam和SGD算法，以及用于条件随机场（CRF）的torchcrf库
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
"""
建立网络模型结构  ==========================================#用于训练或测试模型时初始化模型结构=====================================
"""
#定义了一个名为TorchModel的类，继承自nn.Module，这是所有神经网络模块的基类
class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()#调用父类nn.Module的初始化方法
        #从配置中读取隐藏层大小、词汇表大小和分类数量，并对词汇表大小进行加1操作（通常为了包含特殊的padding token）
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        #创建一个嵌入层，将词汇表中的索引映射为hidden_size维的向量。padding_idx=0指定了索引为0的词向量为全0向量，用于padding
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        #创建一个LSTM层，它是一个双向的循环神经网络层，batch_first=True指定输入和输出的第一维是批次大小
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        #由于LSTM层是双向的，输出的隐藏状态维度是hidden_size * 2，这里创建一个全连接层，用于将LSTM层的输出映射到最终的分类数class_num
        self.classify = nn.Linear(hidden_size * 2, class_num)
        #创建一个CRF层，用于序列标注任务中的标签解码
        self.crf_layer = CRF(class_num, batch_first=True)
        #从配置中读取是否使用CRF层的标志
        self.use_crf = config["use_crf"]
        #创建一个交叉熵损失函数，用于训练时计算损失。ignore_index=-1表示在计算损失时忽略标签为-1的数据（通常用于padding的标签）
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    #定义了模型的前向传播方法。如果提供了target（真实标签），则返回损失值；否则返回预测值
    def forward(self, x, target=None):
        #将输入的词索引通过嵌入层转换为词向量
        x = self.embedding(x)  #input shape:(batch_size, sen_len)
        #将嵌入后的词向量输入到LSTM层
        x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        # 将LSTM层的输出通过全连接层进行分类预测
        predict = self.classify(x)
        # 如果提供了真实标签，计算损失
        if target is not None:
#如果使用CRF层，计算CRF的损失。mask是一个布尔张量，指示哪些标签是有效的
            if self.use_crf:
                mask = target.gt(-1)
                return self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #如果不使用CRF，计算交叉熵损失
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:#如果没有提供真实标签，返回模型的预测结果
            #如果使用CRF层，使用Viterbi算法进行解码并返回最有可能的标签序列
            if self.use_crf:
                return self.crf_layer.viterbi_decode(predict)
            else:#如果不使用CRF，直接返回预测结果
                return predict


def choose_optimizer(config, model):#定义了一个名为choose_optimizer的函数，用于根据配置选择优化器
    #从配置中读取优化器类型和学习率
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    #根据配置返回相应的优化器实例，设置其参数和学习率
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)