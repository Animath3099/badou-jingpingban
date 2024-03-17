import torch
import numpy as np
import torch.nn as nn
from Config import config
from transformers import BertModel,BertForTokenClassification,BertTokenizer
from loader import Dataset

class BertNER(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.bert=BertForTokenClassification.from_pretrained(config['bert_path'],num_labels=config['num_labels'])

    def forward(self,input_ids,mask,type_ids,label=None):
        if label is not None:
            output=self.bert(input_ids=input_ids,attention_mask=mask,token_type_ids=type_ids,labels=label)
        else:
            output = self.bert(input_ids=input_ids, attention_mask=mask, token_type_ids=type_ids)

        return output.loss,output.logits

def get_key(val,my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key

if __name__ == '__main__':
    txt="新华社上海二月二十日电，记者谢金虎"

    tokenizer=BertTokenizer(config['vocab_path'])

    output=tokenizer.encode_plus(txt,max_length=config['max_len'],padding='max_length')

    input_ids, mask, type_ids=output['input_ids'],output['attention_mask'],output['token_type_ids']

    input_ids=torch.LongTensor([input_ids]).cuda()
    mask=torch.LongTensor([mask]).cuda()
    type_ids=torch.LongTensor([type_ids]).cuda()

    model=torch.load('NERbert.pth')
    model=model.cuda()

    _,logits=model(input_ids,mask,type_ids)
    logits=torch.argmax(logits,dim=2)

    print(logits.shape)

    # res=torch.argmax(logits,dim=2)
    # res=res.squeeze()
    # res=res.detach().cpu().numpy().tolist()
    # d=Dataset(config)
    # d.load()
    # dic=d.class_type
    # x=[]
    # for elem in res:
    #     x.append(get_key(elem,dic))
    # print(x)