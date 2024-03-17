# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.classify = nn.Linear(hidden_size, class_num)
        self.pool = nn.AvgPool1d(max_length)
        self.activation = torch.relu
        self.dropout = nn.Dropout(0.1)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

    def forward(self, anchor, positive, negative):
        # 确保 anchor、positive 和 negative 已经是张量

        print(type(anchor), type(positive), type(negative))
        anchor_embedded = self.embedding(anchor)
        positive_embedded = self.embedding(positive)
        negative_embedded = self.embedding(negative)

        anchor_feature = self.activation(self.layer(self.pool(anchor_embedded.transpose(1, 2)).squeeze()))
        positive_feature = self.activation(self.layer(self.pool(positive_embedded.transpose(1, 2)).squeeze()))
        negative_feature = self.activation(self.layer(self.pool(negative_embedded.transpose(1, 2)).squeeze()))

        return anchor_feature, positive_feature, negative_feature





def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)
