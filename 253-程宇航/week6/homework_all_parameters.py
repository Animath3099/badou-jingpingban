import torch
import math
import numpy as np
from transformers import BertModel

"""
手动计算bert整体的参数量，并和bert模型进行对比
"""

bert = BertModel.from_pretrained(r"D:\badou-jingpin\bert-base-chinese", return_dict=False)

vocab_size = 21128      # 词表大小
embedding_size = 768       # 单个词维度
hidden_size = 3072         # 隐藏层维度
max_seq_length = 512       # 句子长度
n = 2                     # 输入句子个数


# embedding层     token_embedding + segment_embedding + position_embedding + layer Norm(embedding_size + embedding_size)

token_embedding = vocab_size * embedding_size
segment_embedding = n * embedding_size
position_embedding = max_seq_length * embedding_size
layer_norm = embedding_size + embedding_size
embedding_param = token_embedding + segment_embedding + position_embedding + embedding_size + embedding_size

# self_attention过程层   weight: embedding_size * embedding_size   bias: embedding_size    Q,K,V三个各算一次
self_attention_param = (embedding_size * embedding_size + embedding_size) * 3

# self_attention输出层   线性层：linear  +    layer_norm层
linear = embedding_size * embedding_size +embedding_size
self_attention_out_param = linear + layer_norm

# feed forward层   两个线性层 + 一个layernorm(归一化层)
linear1 = embedding_size * hidden_size + hidden_size
linear2 = embedding_size * hidden_size + embedding_size
feed_forward_param = linear1 + linear2 + layer_norm

# pooler_out层    wx + b
pooler_out_param = embedding_size * embedding_size + embedding_size

# 总参数即为所有层之和
all_param = embedding_param + self_attention_param + self_attention_out_param + feed_forward_param + pooler_out_param

print("模型总参数是：%d" % sum(p.numel() for p in bert.parameters()))
print("diy模型总参数是：", all_param)

