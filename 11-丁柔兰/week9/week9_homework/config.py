# -*- coding: utf-8 -*-

"""
配置参数信息
"""

# 定义一个名为 Config 的字典，用来存储模型训练和评估时需要的配置参数。
Config = {
    "model_path": "model_output",  # 用于存储训练后模型的路径。
    "schema_path": "ner_data/schema.json",  # 存储数据标签架构的 JSON 文件路径。
    "train_data_path": "ner_data/train",  # 训练数据集的路径。
    "valid_data_path": "ner_data/test",  # 验证数据集的路径。
    "vocab_path": "chars.txt",  # 字符集文件的路径，用于构建模型的词汇表。
    "max_length": 100,  # 输入序列的最大长度。
    "hidden_size": 256,  # 模型隐藏层的大小。
    "num_layers": 2,  # 模型中的层数。
    "epoch": 20,  # 训练过程中的迭代次数。
    "batch_size": 16,  # 每个批次的样本数量。
    "optimizer": "adam",  # 优化器的类型，此处使用 Adam。
    "learning_rate": 1e-3,  # 学习率。
    "use_crf": True,  # 是否在模型中使用条件随机场（CRF）。
    "class_num": 9,  # 类别数量，对应于标签架构中的标签类型数。
    "bert_path": r"E:\work_space\pretrain_models\models-bert-base-chinese"  # 预训练的 BERT 模型路径。
}
