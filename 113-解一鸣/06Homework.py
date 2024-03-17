from transformers import BertModel

model = BertModel.from_pretrained("../bert-base-chinese", return_dict=False)

# 获取模型配置
config = model.config

# 初始化总参数数量
total_params_model = 0

# 1. 嵌入层参数
embedding_params = config.vocab_size * config.hidden_size
total_params_model += embedding_params

# 2. 位置编码参数
position_embedding_params = config.max_position_embeddings * config.hidden_size
total_params_model += position_embedding_params

# 3. Transformer 层参数
for i in range(config.num_hidden_layers):
    # 自注意力参数
    self_attention_params = (3 * config.num_attention_heads * config.hidden_size ** 2) // config.num_attention_heads
    total_params_model += self_attention_params

    # 层归一化参数
    layer_norm_params = 2 * config.hidden_size
    total_params_model += layer_norm_params

# 4. 池化层参数
pooler_params = config.hidden_size * config.hidden_size
total_params_model += pooler_params

# 打印整个模型的总参数数量
print(f"\n整个模型的总参数数量: {total_params_model}")


# 打印嵌入层参数数量
print(f"嵌入层参数数量: {embedding_params}")

# 打印位置编码参数数量
print(f"位置编码参数数量: {position_embedding_params}")

# 打印 Transformer 层参数数量
print(f"自注意力参数: {self_attention_params}")
print(f"层归一化参数: {layer_norm_params}")
print(f"Transformer 层参数数量: {self_attention_params + layer_norm_params}")


