# -*- coding: utf-8 -*-
import torch
from loader_1226 import load_data
from sklearn.metrics import precision_recall_fscore_support
"""
模型效果测试
1226
增加01样本的正确率预测
"""
def tensor_to_numpy(x):
    ''' Need to cast before calling numpy()
    '''
    # return (Variable(x).data).cpu().numpy()
    return x.data.type(torch.DoubleTensor).cpu().numpy()

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果
        self.all_label = []
        self.all_pred_labels = []

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        # 清空上一轮结果
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.all_label = []
        self.all_pred_labels = []
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        # 这里只能处理一个bt的内容，先保存起来，后面统一计算
        assert len(labels) == len(pred_results)
        pred_labels = []
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
            pred_labels.append(tensor_to_numpy(pred_label).tolist())
        labels = tensor_to_numpy(labels).tolist()
        self.all_label += labels
        self.all_pred_labels += pred_labels

        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))

        confusion_matrix = {}
        matches = 0
        if len(self.all_pred_labels) != 0:
            for i in range(len(self.all_label)):
                if self.all_label[i][0] == self.all_pred_labels[i]:
                    matches += 1
                string = str(int(self.all_label[i][0])) + " --> " + str(int(self.all_pred_labels[i]))
                if string in confusion_matrix:
                    confusion_matrix[string] += 1
                else:
                    confusion_matrix[string] = 1
            acc = float(matches) / float(len(self.all_label))
            # print("accuracy:", acc)
            self.logger.info("confusion_matrix[target --> pred]: %s", confusion_matrix)
        avg = precision_recall_fscore_support(
            self.all_label, self.all_pred_labels, average='macro')
        avg_P, avg_R, avg_F = avg[0], avg[1], avg[2]
        self.logger.info("avg_P: {}, avg_R: {}, avg_F：{}".format(round(avg_P, 4), round(avg_R, 4), round(avg_F, 4)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
