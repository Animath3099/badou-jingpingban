# -*- coding: utf-8 -*-
import time
from datetime import datetime

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from export_tools.csv_export import CSVFile
from export_tools.export_dict import ExportHyperParam, keys
from export_tools.hyper_param_input import ModelType, PoolingStyle, LearningRate, Optimizer, BatchSize
from model import choose_optimizer, PairwiseNetwork
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    time_0 = time.time()
    #加载训练数据
    dg, train_data = load_data(config["train_data_path"], config)

    #加载模型
    model = PairwiseNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            a, p, n = batch_data
            loss = model(a, p, n)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    config["training_time"] = time.time() - time_0
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    c_export = CSVFile(f"output/{datetime.now().strftime('%Y_%m%d_%H_%M_%S')}.csv",
                       keys)
    for mt in ModelType:
        Config["model_type"] = mt
        for ps in PoolingStyle:
            Config["pooling_style"] = ps
        #     for lr in LearningRate:
        #         Config["learning_rate"] = lr
        #         for bs in BatchSize:
        #             Config["batch_size"] = bs
        #             for o in Optimizer:
        #               Config["optimizer"] = o
            main(Config)
            v = ExportHyperParam.renew_values(Config)
            c_export.write_in(v)