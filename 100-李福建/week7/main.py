import time

import torch
import os
import random
import os
import numpy as np
import logging
from param_config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = TorchModel(config)

    # 选在优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    for epoch in range(config["epoch"]):
        # 训练模式
        model.train()
        logger.info("epoch %d begin" % (epoch + 1))
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()  # 梯度归零
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)  # 计算loss
            #loss = torch.tensor(loss, dtype=float, requires_grad=True)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    return acc


if __name__ == "__main__":
    main(Config)
