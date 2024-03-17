# -*- coding: utf-8 -*-

import torch

from torch.optim import Adam,SGD

from torchcrf import CRF

from transformers import BertModel

class TorchModel(torch.nn.Module):
    def __init__(self,config):
        super(TorchModel, self).__init__()
        self.use_bert = config['use_bert']
        if self.use_bert:

            self.bert = BertModel.from_pretrained(config['bert_path'])

            self.classify_layer = torch.nn.Linear(self.bert.config.hidden_size,config['class_num'])
        else:
            self.embedding = torch.nn.Embedding(config['vocab_size'],config['hidden_size'],padding_idx=0)
            self.lstm_layer = torch.nn.LSTM(config['hidden_size'],config['hidden_size'],num_layers=config['num_layers'],bidirectional=True,batch_first=True)
            self.classify_layer = torch.nn.Linear(config['hidden_size']*2,config['class_num'])
        self.crf = CRF(config['class_num'],batch_first=True)
        self.use_crf = config['use_crf']
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)


    def forward(self,x,label=None):
        if self.use_bert:
            # print('input_id:',x.shape)
            # print(x.gt(0))
            x = self.bert(x,attention_mask=x.gt(0)).last_hidden_state
            # print('bert out:',bert_out.shape)
            pred = self.classify_layer(x)
        else:
            x = self.embedding(x) # batch_size * max_length * hidden_size
            x,_ = self.lstm_layer(x) # batch_size * max_length * hidden_size
            pred = self.classify_layer(x) #batch_size * max_length * class_num
        if label is not None:
            if self.use_crf:
                mask = label.gt(-1)
                return -self.crf(pred,label,mask=mask,reduction='mean')
            else:

                return self.loss(pred.view(-1,pred.shape[-1]),label.view(-1)) # x:(batch_size*max_length)*class_num  label:(batch_size*max_length)*class_num
        else:
            if self.use_crf:
                return self.crf.decode(pred)
            else:
                return pred
def choose_optimizer(config,model):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']

    if optimizer =='adam':
        return Adam(model.parameters(),lr=learning_rate)
    else:
        return SGD(model.parameters(),lr=learning_rate)