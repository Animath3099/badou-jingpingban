# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
import torch.cuda

from loader import load_data

import re


class Evaluator():
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.valid_data = load_data(self.config['valid_data_path'], self.config,shuffle=False)

    def eval(self, epoch):
        self.logger.info('eval epoch {}'.format(epoch))
        self.model.eval()

        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [item.cuda() for item in batch_data]

            sentences = self.valid_data.dataset.sentences[
                        index * self.config['batch_size']:(index + 1) * self.config['batch_size']]

            input_ids, labels = batch_data

            with torch.no_grad():
                preds = self.model(input_ids)

            # print('pred shape:',preds.shape)
            if not self.config['use_crf']:
                preds = torch.argmax(preds, dim=-1)

            self.write_stats(sentences, preds, labels)

        self.show_stats()

    def write_stats(self, sentences, preds, labels):
        self.stats = {
            'LOCATION': defaultdict(int),
            'ORGANIZATION': defaultdict(int),
            'PERSON': defaultdict(int),
            'TIME': defaultdict(int)
        }

        for sentence, pred, label in zip(sentences, preds, labels):
            # print('sentence:', len(sentence))
            # print('label:', label.gt(-1).sum().item())
            if not self.config['use_crf']:
                pred = pred.cpu().detach().tolist()
            label = label.cpu().detach().tolist()
            pred_entries = self.entry_decode(sentence, pred)
            true_entries = self.entry_decode(sentence,label)
            # print('sentence:',sentence)
            # print('pred:',pred)
            # print('pred_entries:',pred_entries)
            # print('true_entries:',true_entries)


            for key in ['LOCATION','ORGANIZATION','PERSON','TIME']:
                self.stats[key]['find_real_entry']+= len([entry for entry in pred_entries[key] if entry in true_entries[key]])
                self.stats[key]['pred_entry'] += len(pred_entries[key])
                self.stats[key]['true_entry'] += len(true_entries[key])

    def show_stats(self):
        self.logger.info('+---------------------')
        F1_scores = []
        micro_entrys = defaultdict(int)
        for key in ['LOCATION','ORGANIZATION','PERSON','TIME']:

            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats[key]['find_real_entry'] / (self.stats[key]['pred_entry'] +1e-5)
            recall = self.stats[key]['find_real_entry'] / (self.stats[key]['true_entry']+1e-5)
            f1 = 2 * precision * recall / (precision + recall + 1e-5)
            F1_scores.append(f1)
            self.logger.info(f'{key}:准确率:{precision},召回率:{recall},F1:{f1}')
            micro_entrys['find_real_entry'] += self.stats[key]['find_real_entry']
            micro_entrys['pred_entry'] += self.stats[key]['pred_entry']
            micro_entrys['true_entry'] += self.stats[key]['true_entry']

        self.logger.info(f'macro F1:{np.mean(F1_scores)}')


        micro_precision = micro_entrys['find_real_entry'] / (micro_entrys['pred_entry'] +1e-5)
        micro_recall = micro_entrys['find_real_entry'] / (micro_entrys['true_entry']+1e-5)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-5)
        self.logger.info(f'micro F1:{micro_f1}')
        self.logger.info('-+++++++++++++++++++++')



    '''
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    '''
    def entry_decode(self, sentence, label):
        label = ''.join([str(i) for i in label[:len(sentence)]])
        # if '-1' in label:
        #     print('sentence:',sentence)
        #     print('label:',label)

        results = defaultdict(list)
        for location in re.finditer('(04+)',label):
            s,e = location.span()
            results['LOCATION'].append(sentence[s:e])
        for location in re.finditer('(15+)',label):
            s,e = location.span()
            results['ORGANIZATION'].append(sentence[s:e])
        for location in re.finditer('(26+)',label):
            s,e = location.span()
            results['PERSON'].append(sentence[s:e])
        for location in re.finditer('(37+)',label):
            s,e = location.span()
            results['TIME'].append(sentence[s:e])
        return results





if __name__ == '__main__':
    pass
    # 他自任总经理的成都市名优果品开发公司, 拥有几百亩枇杷基地, 近几年虽是销售低谷, 但年利润也有50万元。
    # 778888888888888888888888888888888888888888888888888888888888888
    # 266888888882688888111111111111111111111111111111111111111111111
    # strs = '00112233040000044'
    # for location in re.finditer("(04+)",strs):
    #     s,e = location.span()
    #     print(s,e)
