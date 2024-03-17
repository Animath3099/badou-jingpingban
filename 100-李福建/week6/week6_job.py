"""
计算bert模型可训练参数总数
"""

# embedding层
word_embedding = 21128 * 768
segment_embedding = 2 * 768
position_embedding = 512 * 768
embedding_layerNorm_w = 768  # layerNorm
embedding_layerNorm_b = 768
embedding_param_count = (21128 + 2 + 512) * 768 + 768 * 2
print(embedding_param_count)

# transformer编码层
# self-attention层
q_w = 768 * (768 / 12)  # 除以12是因为多头机制,代表每个头的权重
k_w = 768 * (768 / 12)
v_w = 768 * (768 / 12)
attention_w = 768 * 768
attention_layerNorm_w = 768  # layerNorm
attention_layerNorm_b = 768

# 括号里的12是12个头机制，最后一个12代表transformer12层结构
self_attention_param_count = (768 * (768 / 12) * 3 * 12 + (768 * 768) + 768 * 2) * 12
print(self_attention_param_count)

# Feed-Forward层
output_w = 768 * 3072  # 通过第一个w将维度扩大4倍 => 3072
intermediate_w = 3072 * 768  # 通过此w将维度降至768
feed_forward_layerNorm_w = 768  # layerNorm
feed_forward_layerNorm_b = 768
feed_forward_param_count = (3072 * 768 + 768 * 3072 + 768 * 2) * 12
print(feed_forward_param_count)
total_param_count = embedding_param_count + self_attention_param_count + feed_forward_param_count
print(total_param_count)
