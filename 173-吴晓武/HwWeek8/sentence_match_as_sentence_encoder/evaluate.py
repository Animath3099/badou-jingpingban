# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torch

    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for data in self.train_data:
          data_tensor = data[0]  # Assume data is a tensor
          if len(data_tensor.shape) == 2:
            print("Shape of data tensor:", data_tensor.shape)
            # Extracting indices from tensor
            
            question_indices = torch.unbind(data_tensor, dim=0)
            for index, question_id in enumerate(question_indices):
                self.question_index_to_standard_question_index[len(self.question_ids)] = index
                self.question_ids.append(question_id.tolist())
  # Assuming question_id is a scalar tensor
          else:
            print("Error: Expected a 2D tensor for data")

        with torch.no_grad():
         question_tensor = torch.tensor(self.question_ids, dtype=torch.long, device=self.device)
         self.knwb_vectors = self.model(question_tensor)
         self.knwb_vectors = F.normalize(self.knwb_vectors, dim=-1)



    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            input_ids, labels = [torch.tensor(d, dtype=torch.long) for d in batch_data]  # 转换为张量类型
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)  # 将数据移动到正确的设备
            with torch.no_grad():
              test_question_vectors = self.model(*input_ids)  # 调用模型
            self.write_stats(test_question_vectors, labels)
        self.show_stats()

    def write_stats(self, test_question_vectors, labels):
        test_question_vectors = F.normalize(test_question_vectors, dim=-1)
        for test_question_vector, label in zip(test_question_vectors, labels):
            res = torch.matmul(test_question_vector.unsqueeze(0), self.knwb_vectors.t())
            _, hit_index = torch.max(res, dim=1)
            hit_index = self.question_index_to_standard_question_index[hit_index.item()]
            if int(hit_index) == int(label.item()):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        accuracy = correct / total if total != 0 else 0
        self.logger.info("预测集合条目总量：%d" % total)
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % accuracy)
        self.logger.info("--------------------")
