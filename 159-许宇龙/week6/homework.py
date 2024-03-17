from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"..\..\..\bert-base-chinese", return_dict=False)

# 根据配置参数，计算模型的规模
vocab_size = 21128
num_layers = 1
hidden_size = 768
max_position_embeddings = 512
type_vocab_size = 2  # 一次输入token包含几段话
intermediate_size = 3072 # transformer层中间层参数


# 计算embedding层参数
# ( token + position + segment ) embedding
embed_num = vocab_size * hidden_size + max_position_embeddings * hidden_size + type_vocab_size * hidden_size
layer_norm1 = hidden_size + hidden_size

# 计算transformer层参数
q_k_v_W_b = (hidden_size * hidden_size + hidden_size) * 3
attention_output_num = hidden_size * hidden_size + hidden_size
layer_norm2 = hidden_size + hidden_size
feed_forward = hidden_size * intermediate_size + intermediate_size + hidden_size * intermediate_size + hidden_size
layer_norm3 = hidden_size + hidden_size
transformer_num = (q_k_v_W_b + attention_output_num + layer_norm2 + feed_forward + layer_norm3) * num_layers

# 计算pooler层参数
pooler_num = hidden_size * hidden_size + hidden_size

total_num = embed_num + layer_norm1 + transformer_num + pooler_num
print("diy bert总参数为：%d", total_num)

#统计bert模型参数
# 统计模型参数数量
num_parameters = sum(p.numel() for p in bert.parameters() if p.requires_grad)

print("Total Parameters:", num_parameters)


