# -*- coding: utf-8 -*-

Config = {
    "model_path": "model_output",
    "pretrain_model_path":r"D:\BaiduNetdiskDownload\bert-base-chinese",
    "train_data_path": "train.csv",
    "valid_data_path": "valid.csv",
    "model_type":"LSTM",
    "vocab_path":"chars.txt",
    "max_length": 463,
    "hidden_size": 256,
    "epoch": 5,
    "batch_size": 128,
    "pooling_style":"max",
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": 987,
    "num_layers":1
}