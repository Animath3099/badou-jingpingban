
import torch
import numpy as np


"""
bert 参数量计算
"""

"""
bert 网络结构主要包括以下几个部分：
1、基于embeddings 的词嵌入部分。参数量包括 (21128, 768) + (512, 768) + (2, 768) + (768) * 2
    BertEmbeddings:
      word_embeddings: Embedding(21128, 768, padding_idx=0)
      position_embeddings: Embedding(512, 768)
      token_type_embeddings: Embedding(2, 768)
      LayerNorm: LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      dropout: Dropout(p=0.1, inplace=False)  不改变形状
  
2、基于 encoder 的编码部分( x 12 )。参数量包括 12 x (3 x (768, 768) + (768, 768) + (768) * 2 + (768, 3072) + (3072, 768) + (768) * 2)
        attention:
          BertSelfAttention:
            query: Linear(in_features=768, out_features=768, bias=True)
            key: Linear(in_features=768, out_features=768, bias=True)
            value: Linear(in_features=768, out_features=768, bias=True)
            dropout: 不改变形状
          
          BertSelfOutput:
            dense: Linear(in_features=768, out_features=768, bias=True)
            LayerNorm: LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            dropout: 不改变形状
          
        
        BertIntermediate:
          dense: Linear(in_features=768, out_features=3072, bias=True)
        
        BertOutput:
          dense: Linear(in_features=3072, out_features=768, bias=True)
          LayerNorm: LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          dropout: Dropout(p=0.1, inplace=False)

3、将attention 计算结果再经过线性层和激活曾最后输出。参数量包括 (768, 768)
    BertPooler:
      dense: Linear(in_features=768, out_features=768, bias=True)
      activation: Tanh()

总的参数量：
        (21128, 768) + (512, 768) + (2, 768) + (768) * 2
    +   12 x (3 x (768, 768) + (768, 768) + (768) * 2 + (768, 3072) + (3072, 768) + (768) * 2)
    +   (768, 768)
    =   102137496
"""
