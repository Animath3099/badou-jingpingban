import torch
import os
import random
import numpy as np
import logging
from config import Config
from loader import load_data
from model import TorchModel, choose_optimizer
from export_to_csv import export2csv
from evaluate import Evaluator



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
训练程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # load_train_data
    train_data = load_data(config["train_data_path"], config)
    
    # optional model
    optional_model = ["RNN", "CNN", "LSTM"]
    
    # output_data
    test_data = []
    
    # load_model
    for c_model in optional_model:
        config["model_type"] = c_model
        model = TorchModel(config)
        out_data = []
        cuda_flag = torch.cuda.is_available()
        if cuda_flag:
            logger.info("cuda is available")
            model.cuda()
        optimizer = choose_optimizer(config, model)
        evaluator = Evaluator(config, model, logger)
        logger.info("Current model type: "+c_model)
        # train
        time_start = time.time()
        for epoch in range(config["epoch"]):
            epoch += 1
            model.train()
            logger.info("epoch %d begin" % epoch)
            train_loss = []
            for index, batch_data in enumerate(train_data):
                if cuda_flag:
                    batch_data = [d.cuda() for d in batch_data] 
                           
                optimizer.zero_grad()
                input_ids, labels = batch_data
                loss = model(input_ids, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item()) 
                if index % int(len(train_data) / 2) == 0:
                    logger.info("batch loss: %f" % loss)
            logger.info("epoch average loss: %f" % np.mean(train_loss))
            acc = evaluator.eval(epoch)
        
        time_end = time.time()
        time_cost = time_end - time_start
        keys = config.keys()
        for key in keys:
            value = config[key]
            out_data.append(value)
        out_data.extend((np.mean(train_loss), acc, time_cost))
        test_data.append(out_data)
    return acc, test_data

if __name__ == "__main__":
    import time
    
    acc, test_data = main(Config)
    
    export2csv(test_data)