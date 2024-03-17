from transformers import BertModel

# 加载BERT模型
model = BertModel.from_pretrained(r'E:\\data\\hub\\bert_base_chinese')

# 计算模型的可训练参数量
num_params = model.num_parameters()

print("BERT模型可训练参数量：", num_params)