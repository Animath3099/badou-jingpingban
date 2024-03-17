

# embedding
word_embeddings = 21128 * 768
posting_embeddings = 512 * 768
segment_embeddings = 2 * 768
e_w = 768
e_b = 768
embedding_all_w = word_embeddings + posting_embeddings + segment_embeddings + e_w + e_b
print(embedding_all_w)


# self_attention
# w + b
q = 768 * 768 + 768
k = 768 * 768 + 768
v = 768 * 768 + 768
s_w = 768 * 768
s_b = 768
# 12层self_attention
self_attention_all_w = (q + k + v + s_w + s_b) * 12
print(self_attention_all_w)

# self_attention之后bn
layer1_w = 768
layer1_b = 768
layer1_all_w = layer1_w + layer1_b
print(layer1_all_w)

# feed_forward 两层layer 一层扩大 一层缩小 增加可训练参数
i_w = 3072 * 768
i_b = 3072
o_w = 3072 * 768
o_b = 768
feed_forward_all_w = i_w + i_b + o_w + o_b
print(feed_forward_all_w)

# feed_forward之后bn
layer2_w = 768
layer2_b = 768
layer2_all_w = layer2_w + layer2_b

all_w = embedding_all_w + self_attention_all_w + layer1_all_w + feed_forward_all_w + layer2_all_w
print(all_w)
