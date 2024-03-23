# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

"""
建立网络模型结构
"""


class LanguageModel(nn.Module):
    def __init__(self,   config):
        super(LanguageModel, self).__init__()
        self.layer = BertModel.from_pretrained(config.pretrain_model_path,  return_dict=False, local_files_only=True)
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)
        self.classify = nn.Linear(config.hidden_size, self.tokenizer.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        x = self.layer(x)  # output shape:(batch_size, sen_len, input_dim)
        if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        elif not isinstance(x, torch.Tensor):
            x = x[0]
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)

        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


def choose_optimizer(config, model):
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    model = LanguageModel( Config)
    print(model)
