import  numpy as np
import torch
from torch.utils.data import DataLoader
from Config import config
from collections import defaultdict
from transformers import BertTokenizer

def get_key(val,my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key

class Dataset(torch.utils.data.Dataset):
    def __init__(self,config):
        self.config=config

        self.train_sentences=[]  #[ sentence1,sentence2,...]
        self.train_labels=[] #[ [BEOS,],[BEOS]...]

        self.test_sentences=[]
        self.test_labels=[]


        self.class_type = defaultdict(int)
        self.mode='train'
        self.tokenizer=BertTokenizer(config['vocab_path'])

        self.load()

    def load(self):
        self.mode='train'
        sentences=[]
        labels=[]
        with open(self.config['train_path'], 'r',encoding='utf8') as f:
            sentence=''
            label=[]
            for line in f:
                line=line.strip()
                line=line.split(' ')
                if len(line)>1:
                    sentence+=line[0]
                    label.append(line[1])
                    self.class_type[line[1]]=self.class_type.get(line[1],len(self.class_type))
                else:
                    sentences.append(sentence)
                    labels.append(label.copy())
                    sentence=''
                    label.clear()
        f.close()
        self.train_sentences=sentences
        self.train_labels=labels

    def load_test(self):
        self.mode='test'
        sentences = []
        labels = []
        with open(self.config['valid_path'], 'r', encoding='utf8') as f:
            sentence = ''
            label = []
            for line in f:
                line = line.strip()
                line = line.split(' ')
                if len(line) > 1:
                    sentence += line[0]
                    label.append(line[1])
                    self.class_type[line[1]] = self.class_type.get(line[1], len(self.class_type))
                else:
                    sentences.append(sentence)
                    labels.append(label.copy())
                    sentence = ''
                    label.clear()
        f.close()
        self.test_sentences = sentences
        self.test_labels = labels

    def __getitem__(self, index):
        if self.mode=='train':
            #取出句子和标签
            txt=self.train_sentences[index]
            label=self.trans_label(self.train_labels[index])

            #编码
            input_ids, attention_mask, input_type =self.encoder(txt)
            #label补齐到token长度
            label = self.padding_label(txt, label)

            res= [torch.LongTensor(input_ids),torch.LongTensor(attention_mask),torch.LongTensor(input_type),
                    torch.LongTensor(label)]
            return res
        elif self.mode=='test':
            txt = self.test_sentences[index]
            label = self.trans_label(self.test_labels[index])

            # 编码
            input_ids, attention_mask, input_type = self.encoder(txt)
            # label补齐到token长度
            label = self.padding_label(txt, label)

            res = [torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(input_type),
                   torch.LongTensor(label)]
            return res

    def __len__(self):
        if self.mode=='train':
            return len(self.train_sentences)
        elif self.mode=='test':
            return len(self.test_sentences)

    def encoder(self,txt):
        '''
        编码字符串
        :param txt:句子，汉字形式字符串
        :return: 编码，mask，type_ids
        '''
        #tokenize
        encode_output=self.tokenizer.encode_plus(txt,truncation=True,max_length=config['encode_dim'],
                                                 padding='max_length')

        input_ids,attention_mask,input_type=encode_output['input_ids'],encode_output['attention_mask'],\
            encode_output['token_type_ids']
        return input_ids,attention_mask,input_type

    def padding_label(self,txt,label):
        if len(label)>=config['encode_dim']-2:
            label=label[:config['encode_dim']-2]
            label.insert(0,self.class_type['O'])
            label.append(self.class_type['O'])
            return label

        #插入['end]对应
        label.append(self.class_type['O'])
        #插入[CLS]对应label
        label.insert(0,self.class_type['O'])
        label.extend([self.class_type['O']]*(config['encode_dim']-len(label)))
        return label

    def trans_label(self,label_list):
        labels=[]
        for label in label_list:
            labels.append(self.class_type[label])
        return labels

if __name__ == '__main__':
    d=Dataset(config)
    dataloader=DataLoader(d,batch_size=config['batch_size'],shuffle=True)
    d.load_test()
    for i,b in enumerate(dataloader):
        print(1)