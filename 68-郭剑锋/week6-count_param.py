import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(r"D:\masterPeoject\nlp\week6 语言模型和预训练\bert-base-chinese", return_dict=False)

n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数


#embedding参数  词向量+位置+segment+LN层
em_para = vocab*embedding_size + max_sequence_length*embedding_size + n*embedding_size +embedding_size*2

#transformer参数  self_attention+LN+两层MLP+LN    经过self_attention后会经过一层线性层再整合一次注意力输出
transformers_para = 3*(embedding_size*embedding_size+embedding_size) + embedding_size*embedding_size+embedding_size+ 2*embedding_size + embedding_size*hide_size + hide_size + embedding_size*hide_size + embedding_size + 2*embedding_size

#pool参数 一层MLP
pool_para = embedding_size*embedding_size + embedding_size

all_para = em_para + transformers_para + pool_para

print("我的参数："+ str(all_para))
print("实际参数："+str(sum(p.numel() for p in model.parameters())))