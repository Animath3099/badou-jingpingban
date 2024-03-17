import torch
import math
import numpy as np
from transformers import BertModel

bert = BertModel.from_pretrained(r"E:\学习资料_summary\八斗课程-精品班\第六周\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
print([p.numel() for p in bert.parameters()])
print(sum(p.numel() for p in bert.parameters()))

n=2 # input 句子数
VocabNum = 21128 # 词表的词量
EmbeddingSizeNum = 768 # 词的向量数
MaxLength = 512 # 句子最大长度
Hide_Size = 3072

# embedding 中的参数
# Token Embedding: VocabNum * EmbeddingSizeNum
# Segment Embedding: MaxLength * EmbeddingSizeNum
# Position Embedding: n * EmbeddingSizeNum
embedding_parameter = VocabNum * EmbeddingSizeNum + MaxLength * EmbeddingSizeNum + n * EmbeddingSizeNum
# 加入layer_norm，其中weight768，bias768
embedding_parameter += EmbeddingSizeNum * 2

# self-attention: Q/K/V = EmbeddingSizeNum * EmbeddingSizeNum+EmbeddingSizeNum
self_attention_parameter = (EmbeddingSizeNum * EmbeddingSizeNum+EmbeddingSizeNum) * 3
# 加入layer_norm，此处引入残差机制 embedding+attention，但是不引入parameter
self_attention_parameter += EmbeddingSizeNum * EmbeddingSizeNum+EmbeddingSizeNum + EmbeddingSizeNum * 2

# Feed Forward: 先放大到3072，再缩小到768
feed_forward = EmbeddingSizeNum * Hide_Size + Hide_Size + EmbeddingSizeNum * Hide_Size + EmbeddingSizeNum + EmbeddingSizeNum * 2

# FC
FC_parameter = EmbeddingSizeNum * EmbeddingSizeNum+EmbeddingSizeNum

print(embedding_parameter,self_attention_parameter,feed_forward,FC_parameter)
print(embedding_parameter+self_attention_parameter+feed_forward+FC_parameter)

