import math


def count_parameters(hidden_size, vocab_size, max_position_embeddings, type_vocab_size, intermediate_size):
    # 词嵌入层参数量
    embedding_params = vocab_size * hidden_size + max_position_embeddings * hidden_size + type_vocab_size * hidden_size

    # 自注意力层参数量
    # 每个Transformer层有3个权重矩阵（Q, K, V）和1个输出权重矩阵
    attention_params_per_layer = (3 * (hidden_size * hidden_size) + hidden_size + hidden_size) * 4  # 包括权重和偏置
    attention_params = attention_params_per_layer * 12  # 假设有12个注意力头

    # 前馈网络层参数量
    ffn_params_per_layer = (hidden_size * intermediate_size + intermediate_size * hidden_size) * 2  # 包括权重和偏置
    ffn_params = ffn_params_per_layer * 12  # 假设有12个Transformer层

    # 池化层参数量
    pooler_params = hidden_size * hidden_size + hidden_size

    # 总参数量
    total_params = embedding_params + attention_params + ffn_params + pooler_params

    return total_params


# 假设的参数值
hidden_size = 768
vocab_size = 21128  # 根据BERT-base中文模型的词汇表大小
max_position_embeddings = 512
type_vocab_size = 2
intermediate_size = hidden_size * 4  # 通常为hidden_size的4倍

# 计算总参数量
total_params = count_parameters(hidden_size, vocab_size, max_position_embeddings, type_vocab_size, intermediate_size)
print(f"Total number of parameters: {total_params}")