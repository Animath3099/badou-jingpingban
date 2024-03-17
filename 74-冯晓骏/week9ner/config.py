# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",

    "seed": 987,
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "optimizer": "adam",
    "learning_rate": 5e-6,
    "epoch": 20,
    "batch_size": 16,
    "hidden_size": 256,
    "num_layers": 2,
    "class_num": 9,
    "use_crf": True,

    "use_bert":True,
    "bert_path":r'D:/work/bert'
}