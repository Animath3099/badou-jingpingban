# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "pretrain_model_path": 'C:/Users/Admin/bert-base-chinese',
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train.txt",
    "valid_data_path": "ner_data/test.txt",
    "vocab_path": "chars.txt",
    "max_length": 25,
    "hidden_size": 768,
    "epoch": 30,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9
}