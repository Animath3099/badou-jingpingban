# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "log_file": "log_file",
    "class_num": 2,
    "model_path": "output",
    # "new_train_data_path": "../data/Sentiment_classification_data_processing/new_train_data.json",
    "train_data_path": "../data/Sentiment_classification_data_processing/new_train_data.json",
    "valid_data_path": "../data/Sentiment_classification_data_processing/test_data.json",
    "vocab_path": "chars.txt",
    "model_type": "rnn",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 16,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path": r"E:\data\hub\bert_base_chinese",
    "seed": 987
}
