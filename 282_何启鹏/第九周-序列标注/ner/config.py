# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "bert_requires_grad": False,
    "path_log_loss":"model_output/loss_file/bert",
    "use_bert": False,
    "pretrain_model_path": r"E:\data\hub\bert_base_chinese",
    "vocab_path": r"E:\data\hub\bert_base_chinese\vocab.txt",
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "max_length": 50,
    "hidden_size": 768,
    "epoch": 10,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": None
}