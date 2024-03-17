config={
    'model':"edit_match",
    'lr':0.001,
    "optimizer":"Adam",
    "epoch":100,
    "batch_size":32,
    "max_len":30,
    "embedding_dim":128,
    'hidden_dim':128,
    "train_path":r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\train.json",
    "valid_path":r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\valid.json",
    "schema_path":r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\schema.json",
    "positive_rate":0.5,#正负样本比例
    "epoch_data_size": 1000,  # 每轮训练中采样数量
    "out_dim":128,
    "vocab_path":r"E:\python\bert_file\vocab.txt"
}
