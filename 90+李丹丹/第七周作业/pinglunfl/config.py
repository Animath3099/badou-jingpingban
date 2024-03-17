# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../wbfl/文本分类训练集.csv",
    "valid_data_path": "../wbfl/验证集.csv",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 100,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 128,
    "pooling_style": "avg",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"C:\Users\28194\PycharmProjects\pythonProject2\week6\week6 语言模型和预训练\下午\bert-base-chinese",
    "seed": 987
}
