# -*- coding: utf-8 -*-

import torch

import random

import numpy as np

from config import Config

import logging

from loader import load_data

from model import TorchModel,choose_optimizer

from evaluate import Evaluator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 设置随机种子
seed = Config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def main(config):
    # 加载数据
    train_data = load_data(config['train_data_path'],config)

    # 构建模型
    model = TorchModel(config)

    #GPU tag
    cuda_available =  torch.cuda.is_available()
    if cuda_available:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()


    # 设置优化器
    optimizer = choose_optimizer(config,model)

    # 设置评估器
    evaluator = Evaluator(model,config,logger)

    # 测试模型
    for epoch in range(config['epoch']):

        #开始训练
        model.train()

        logger.info("Epoch: {}".format(epoch+1))

        #记录loss
        train_loss = []

        for index,batch_data in enumerate(train_data):

            #梯度归零
            optimizer.zero_grad()

            input_ids,labels = batch_data

            if cuda_available:
                input_ids = input_ids.cuda()
                labels = labels.cuda()
            # print(input_ids,labels)

            loss = model(input_ids,labels)

            train_loss.append(loss.item())

            #反向传播
            loss.backward()

            #更新参数
            optimizer.step()

        evaluator.eval(epoch+1)

        logger.info('Train loss: {}'.format(np.mean(train_loss)))

if __name__ == '__main__':
    main(Config)