# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path":"../chars.txt",
    "max_length": 45,
    "hidden_size": 256,
    "epoch": 100,
    "batch_size": 16,
    "epoch_data_size": 400,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "loss_type":"triple_loss",
    "bert":True,
    "pretrain_model_path":r"F:\BaiduNetdiskDownload\八斗课程-精品班\第六周\bert-base-chinese\bert-base-chinese",
    "seed": 987,
    "layer_hidden_size":256
    
}