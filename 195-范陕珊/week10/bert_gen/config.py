# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import json


class _Config:
    def __init__(self, data):
        self.model_path = data["model_path"]
        self.corpus = data["corpus"]
        self.max_length = data["max_length"]
        self.hidden_size = data["hidden_size"]
        self.kernel_size = data["kernel_size"]
        self.num_layers = data["num_layers"]
        self.epoch = data["epoch"]
        self.train_sample = data['train_sample']
        self.batch_size = data["batch_size"]
        self.pooling_style = data["pooling_style"]
        self.optimizer = data["optimizer"]
        self.learning_rate = data["learning_rate"]
        self.pretrain_model_path = data["pretrain_model_path"]
        self.seed = data["seed"]
        self.window_size = data["window_size"]


Config = _Config(json.load(open("config.json")))
print(Config.seed)
