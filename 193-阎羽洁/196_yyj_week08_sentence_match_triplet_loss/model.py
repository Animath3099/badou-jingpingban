# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

from sentence_match_as_sentence_encoder.model_table import CNN, GatedCNN, StackGatedCNN, RCNN, BertLSTM, BertCNN, \
    BertMidLayer

"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        model_type = config["model_type"]
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        num_layers = config["num_layers"]
        max_length = config["max_length"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.classify = nn.Linear(hidden_size, 2)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.CrossEntropyLoss()

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()
        return x

class PairwiseNetwork(nn.Module):
    def __init__(self, config):
        super(PairwiseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.triplet_loss = nn.TripletMarginLoss()
        # self.cosine_loss = nn.CosineEmbeddingLoss()

    # # 计算余弦距离  1-cos(a,b)
    # # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    # def cosine_distance(self, tensor1, tensor2):
    #     tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
    #     tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
    #     cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
    #     return 1 - cosine
    #
    # def cosine_triplet_loss(self, a, p, n, margin=None):
    #     ap = self.cosine_distance(a, p)
    #     an = self.cosine_distance(a, n)
    #     if margin is None:
    #         diff = ap - an + 0.1
    #     else:
    #         diff = ap - an + margin.squeeze()
    #     return torch.mean(diff[diff.gt(0)]) #greater than

    def forward(self, a, p=None, n=None):
        if p is not None and n is not None: #输入同类的句子 -》 计算余弦距离
            a = self.sentence_encoder(a)
            p = self.sentence_encoder(p)  # vec:(batch_size, hidden_size)
            n = self.sentence_encoder(n)
            return self.triplet_loss(a, p, n)
        else:
            a = self.sentence_encoder(a)
            return a

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = PairwiseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l_1 = torch.LongTensor([[2,2,3,4], [4,2,3,4]])
    y_1 = model(s1, s2, l_1)
    print(y_1)
