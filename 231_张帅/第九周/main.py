# -*- coding: utf-8 -*-
from datetime import datetime

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data


def generate_logger():
    # 获取当前日期，格式化为"yyyyMMdd"
    current_date = datetime.now().strftime("%Y%m%d")
    # 定义文件路径
    file_path = f"logs/{current_date}.log"
    # 检查目录是否存在，如果不存在则创建
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 检查文件是否存在，如果不存在则创建
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            pass  # 创建空文件
    logger = logging.getLogger(__name__)
    # [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    logger.setLevel(logging.INFO)
    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=file_path, encoding='utf-8')  # 指定日志文件的路径和名称
    # 再创建一个formatter，用于设定日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(file_handler)
    return logger

logger = generate_logger()

"""
模型训练主程序
"""

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
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    try:
        model, train_data = main(Config)
    except Exception as e:
        logger.error(e,exc_info=True)
