# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "pretrain_model_path": r"D:\e_Study\NLP\f_pretrained_models\bert-base-chinese",
    "vocab_path":"../chars.txt",
    "vocab_size": None,
    "model_type": "bert",
    "pooling_style": "max",
    "max_length": 20,
    "kernel_size": 3,
    "hidden_size": 128,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 32,
    "epoch_data_size": 1000,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "accuracy": None,
    "training_time": None,
    "acc_round_1": None,
    "acc_round_2": None,
    "acc_round_3": None,
    "acc_round_4": None,
    "acc_round_5": None,
    "acc_round_6": None,
    "acc_round_7": None,
    "acc_round_8": None,
    "acc_round_9": None,
    "acc_round_10": None,
    "acc_round_11": None,
    "acc_round_12": None,
    "acc_round_13": None,
    "acc_round_14": None,
    "acc_round_15": None,
}