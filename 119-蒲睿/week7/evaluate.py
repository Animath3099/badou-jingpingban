# -*- coding: utf-8 -*-
import torch
from loader import load_data

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}
        
    def eval(self, epoch):
        self.logger.info("Epoch %d Evaluating model..." % epoch)
        self.model.eval()
        self.stats_dict = {"correct":0, "wrong":0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                predicted_results = self.model(input_ids)
            self.write_stats(labels, predicted_results)
        acc = self.show_stats()
        return acc
    
    def write_stats(self, labels, predicted_labels):
        assert len(labels) == len(predicted_labels)
        for label, predicted_label in zip(labels, predicted_labels):
            true_label = torch.argmax(predicted_label)
            if int(true_label) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return
    
    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集总条目 %d " % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("-------------------")
        return correct / (correct + wrong)