# -*- coding: utf-8 -*-

import torch
import random
import csv
import os
import numpy as np
import time
import logging
from luowulin_62.week7.config import Config
from luowulin_62.week7.model import TorchModel, choose_optimizer
from luowulin_62.week7.evaluate import Evaluator
from luowulin_62.week7.loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
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
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc
class ClassifivationResunlt:
    def __init__(self, name, time, acc):
        self.name = name
        self.time = time
        self.acc = acc
if __name__ == "__main__":
    list_model = ["cnn", "lstm","gru","rnn"]
    list_result = list()
    list_result_head = ["Model", "准确率acc", "耗时time"]
    list_result.append(list_result_head)
    for model in list_model:
        list_result_temp = list()
        list_result_temp.append(model)
        start_time = time.time()
        Config["model_type"] = model
        acc = round(main(Config), 5)
        list_result_temp.append(acc)
        total_time = round(time.time() - start_time, 5)
        list_result_temp.append(total_time)
        print("最后一轮准确率：", acc, "当前配置：", Config["model_type"])
        list_result.append(list_result_temp)
    with open("data/classification_result.csv", 'w',encoding='utf-8',newline='') as file:
        writer = csv.writer(file)
        for row in list_result:
            writer.writerow(row)
        file.close()
