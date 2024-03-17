# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SentenceEncoder, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

def main(config):
 
    print("当前工作目录 : %s" % os.getcwd())

# 如果需要更改工作目录，可以使用 os.chdir
# os.chdir('您的目标工作目录路径')

    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = SentenceEncoder(config)
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
       
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for batch_data in train_data:
             optimizer.zero_grad()
             if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
             anchor, positive, negative = batch_data
    # 计算三元组损失
             anchor_feature, positive_feature, negative_feature = model(anchor, positive, negative)
    # 计算三元组损失
             loss = model.triplet_loss(anchor_feature, positive_feature, negative_feature)
    # 记录损失值
             train_loss.append(loss.item())
    # 反向传播和参数更新
             loss.backward()
             optimizer.step()
    logger.info(f"epoch average loss: {np.mean(train_loss):.4f}")
    evaluator.eval(epoch + 1)
    model_path = os.path.join(config["model_path"], f"epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main(Config)