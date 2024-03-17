import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as pyl
import json

"""
    编写一个简单的机器学习任务
    通过判断一个字符中 a 字符出现的位置进行分类
    默认是判断一个长度为6的字符,为6分类任务
"""
 #1.创建词表
def build_vocab():
    vocab={"pad":0}
    vocab_str="abcdefghijklmnopqrstuvwxyz"
    index=0
    for char in vocab_str:
        index+=1
        vocab[char]=index
    vocab["unk"]=len(vocab_str)+1
    return vocab


# print(build_vocab())     

 #2.创建训练所需数据集
def build_sample(str_length,vocab):
    str=[]
    for i in range(str_length):
        char=random.choice(list(vocab.keys()))
        str.append(char)
    if set("a")& set(str):
        pass
    else:
        str[random.randint(0,str_length-1)]="a"
    #按照字符a出现的位置进行分类
    y=str.index("a")
    #将字符转换为序列，以备用于embedding层
    x=[vocab.get(char,vocab["unk"]) for char in str]
    return x,y



def build_dataset(batch_size,str_length,vocab):
    X=[]
    Y=[]
    for i in range(batch_size):
        x,y=build_sample(str_length,vocab)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X),torch.LongTensor(Y)

# vocab=build_vocab()
# print(build_dataset(2,6,vocab))
            
#3.创建torch模型
class NLPMultClassify(nn.Module):
    def __init__(self,str_dim,input_size,hidden_size,vocab):
        super(NLPMultClassify,self).__init__()
        self.embedding=nn.Embedding(len(vocab),str_dim)  #Embedding层
        self.rnn=nn.RNN(input_size,hidden_size,batch_first=True) #RNN层 ,RNN包含了全连接层级pooling层的功能
        self.loss=nn.functional.cross_entropy  #交叉熵  因为交叉熵中包含softmax函数，所以省略了激活函数层
    
    def forward(self,x,y=None):
        x=self.embedding(x)
        y_out_put,y_pred=self.rnn(x)
        y_pred = y_pred.squeeze()        
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred
                

#4.模型训练
def main():
    #参数初始化
    epoch_num=20  #训练轮数
    total_num=5000   #每轮训练笔数
    batch_size=20    #batch大小
    str_length=6     #字符长度
    str_dim=20       #每个字符向量长度
    input_size=20
    hidden_size=6
    #创建词表
    vocab=build_vocab()
    #模型创建
    model= NLPMultClassify(str_dim,input_size,hidden_size,vocab)          
    learning_rate=0.0005   #学习率
    #设定优化器
    optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
    log=[]  #画图
    
    #开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        for batch in range(int(total_num/batch_size)):
            dataset_x,dataset_y=build_dataset(batch_size,str_length,vocab) #训练数据
            optim.zero_grad()  #梯度归零
            # print(dataset_x)
            # print(dataset_y)
            loss=model(dataset_x,dataset_y)  #计算损失值
            loss.backward()   #梯度计算
            optim.step()   #权重更新
            watch_loss.append(loss.item())
        print("==========第%d轮的平均loss%f===="%(epoch+1,np.mean(watch_loss)))
        acc=evaluate(model,str_length,vocab)
        log.append([acc,np.mean(watch_loss)])
    #   画图
    pyl.plot(range(len(log)),[l[0] for l in log],label="acc")
    pyl.plot(range(len(log)),[l[1] for l in log],label="loss") 
      
    pyl.legend()
    pyl.show()
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    torch.save(model.state_dict(),"NLPMultClassify.pth")
    return
        
            
    
    
 
 
#5.测试本轮模型
def evaluate(model,str_length,vocab):
    model.eval()
    eval_x,eval_y=build_dataset(200,str_length,vocab)
    y_pred=model(eval_x)
    correct ,wrong= 0,0
    with torch.no_grad():
        for y_p,y_t in zip(y_pred,eval_y):
            if torch.argmax(y_p)==y_t:
                correct+=1
            else:
                wrong+=1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)
    
    
    
 
#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = NLPMultClassify(20,20,6,vocab)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        print(input_string)
        print(result[0])
        print(torch.argmax(result[0]))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" %(input_string, torch.argmax(result[i]), result[i])) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["favfee", "wzsafg", "aqwdeg", "nkwwwa"]
    predict("NLPMultClassify.pth", "vocab.json", test_strings)
 
 










