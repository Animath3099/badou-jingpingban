import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from config  import config
import json
from loader import load_qa,load_vocab,R_loader,sim_loader,triple_loader
from units import process,padding,process_virsion
from evalute import evalute_model
from model_def import Edit_match,Representation,Match_layer,Interaction_layer
import torch
from torch.utils.data import DataLoader
import transformers
transformers.logging.set_verbosity_error()
vocab=load_vocab()


def train_model(model,model_type,config,data_loader=None,if_triple=False):
    if model_type!="edit":
        if config['optimizer']=="SGD":
            optimizer=optim.SGD(model.parameters(),lr=config['lr'])
        elif config['optimizer']=="Adam":
            optimizer=optim.Adam(model.parameters(),lr=config['lr'])

    if model_type=="interaction":
        assert data_loader != None, "没有输入数据加载器"
        train_loss=[]
        for e in range(config['epoch']):
            model.train()
            for index,batch_data in enumerate(data_loader):
                optimizer.zero_grad()
                concat_text,label=batch_data
                loss=model(concat_text,label)
                loss.backward()
                optimizer.step()
                print("epoch:{}  index:{}  loss:{}".format(e,index,loss.item()))
                train_loss.append(loss.item())
        torch.save(model,"interation_model.pth")
        plt.plot(train_loss)
        plt.show()

    if model_type=="repre":
        if if_triple==True:
            train_loss=[]
            for e in range(config['epoch']):
                model.train()
                for index, batch_data in enumerate(data_loader):
                    optimizer.zero_grad()
                    ps1,ps2,ns = batch_data
                    #计算triplet
                    ps1,ps2,ns=model(ps1),model(ps2),model(ns)
                    loss=model.triplet_loss(1,ps1,ps2,ns)
                    loss.backward()
                    optimizer.step()
                    print("Epoch: {}, Batch: {}, Loss: {}".format(e, index, loss.item()))
                    train_loss.append(loss.item())
            plt.plot(train_loss)
            plt.show()
            torch.save(model, "triplet_model.pth")
        else:
            assert  data_loader!=None,"没有输入数据加载器"
            train_loss = []
            for e in range(config['epoch']):
                model.train()
                for index, batch_data in enumerate(data_loader):
                    optimizer.zero_grad()
                    input_id1, input_id2, labels = batch_data
                    loss = model(input_id1, input_id2, labels)
                    train_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    print("Epoch: {}, Batch: {}, Loss: {}".format(e, index, loss.item()))
                    train_loss.append(loss.item())
            plt.plot(train_loss)
            plt.show()
            torch.save(model, "repre_model.pth")

    if model_type=="edit":
        print("统计模型不需要训练")
        valid_match()

def valid_match():
    data=load_qa()
    data.load_train()
    data.load_valid_data()
    model=Edit_match(data.match_pool)
    evalute_model(model,data.valid_pool,algorithm="edit")

def valid_representation(config,vocab):
    '''
    执行较快，在实际查询中，只需要对输入文本做向量化
    :param config:
    :param vocab:
    :return:
    '''
    data = R_loader(config, vocab)
    data.load_data(config['train_path'])
    data_loader = DataLoader(dataset=data, batch_size=config['batch_size'], shuffle=True)

    data.load_data(config['valid_path'])
    data_loader1 = DataLoader(dataset=data, batch_size=1, shuffle=True)

    model1 = torch.load("triplet_model.pth")
    model2=torch.load("repre_model.pth")
    evalute_model(model1, data_loader1, "represent", data.class_questions)
    evalute_model(model2, data_loader1, "represent", data.class_questions)
    return

def valid_interaction(config,vocab):
    '''
    测试时间较长，因为需要每次计算都需要将两个输入送入模型
    :param config:
    :param vocab:
    :return:
    '''
    dataset = sim_loader(config)
    dataset.load(config['train_path'])
    dataset.load(config['valid_path'])
    dataloader_train = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True)
    model = torch.load("interation_model.pth")
    evalute_model(model, dataset.valid_data, algorithm="interaction", class_questions=dataset.class_questions,
                  dataset=dataset)
    return



if __name__ == '__main__':

    dataset=triple_loader(config)
    dataset.load_data(config['train_path'])

    model=Match_layer(vocab,config)

    data_loader=DataLoader(dataset=dataset,batch_size=config['batch_size'],shuffle=True)

   #train_model(model,"repre",config,data_loader,True)

    #表示型网络，包括余弦损失函数和triplet损失函数
    valid_representation(config,vocab)

    #交互型网络
    valid_interaction(config,vocab)

    #编辑距离
    valid_match()

