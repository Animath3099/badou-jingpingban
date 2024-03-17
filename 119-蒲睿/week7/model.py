# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.model_type = config["model_type"]
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "LSTM":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "RNN":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        elif model_type == "CNN":
            self.encoder = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=0)
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
            
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy
        
    def forward(self, x, target=None):
        if self.use_bert:
            # 输入为(batch_size, max_len, embedding_size)
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            if self.model_type == "CNN":
                x = x.transpose(1,2) # 如果是CNN网络，正确语句表意应该为:(batch_size, embedding_size, max_len)
                # Conv1d中的in_channels代表输入通道数即embedding_size, out_channels是卷积后向量维度, seq_len = max_len - kernel_size + 1
                # padding默认为0， 设置为1会默认在每个channel的左右两边补0，此时输入的seq_len变为max_len + 2，输出seq_len = max_len - kernel_size + 3
            x = self.encoder(x)
        
        if isinstance(x, tuple):
            x = x[0]
        
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        if self.model_type == "CNN":
            x = self.pooling_layer(x).squeeze()
        else:
            x = self.pooling_layer(x.transpose(1,2)).squeeze()
        
        predict = self.classify(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict
        
            

class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x
    
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        return SGD(model.parameters(), lr=learning_rate)
        