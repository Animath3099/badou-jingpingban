import torch
from config import config
import jieba
import random
import re
def process(match_pool,vocab,model_type,max_len,target_list=None):
    if model_type=="edit":
        return


    if model_type=="representation":
        data=[]
        count=0
        for q_a in match_pool:
            #去除停用词
            q_a[0]=remove_stop(q_a[0])
            questions,answer=q_a[0],q_a[1]
            for question in questions:

                neg_answer=chose_one(target_list,answer)

                neg_answer_vector=[vocab.get(word,0) for word in neg_answer]
                question_vector=[vocab.get(word,0) for word in question]
                answer_vector=[vocab.get(word,0) for word in answer]

                #padding到标准长度
                question_vector=padding(question_vector,max_len)
                answer_vector=padding(answer_vector,max_len)
                neg_answer_vector=padding(neg_answer_vector,max_len)

                question_vector=torch.LongTensor(question_vector)
                answer_vector=torch.LongTensor(answer_vector)
                neg_answer_vector=torch.LongTensor(neg_answer_vector)


                #对每一对q a，构造一个负样本

                data.append([question_vector,answer_vector,1])
                data.append([question_vector,neg_answer_vector,0])

        return data

def padding(data_list,max_len):
    if len(data_list)>max_len:
        data_list=data_list[:max_len]
    elif len(data_list)<max_len:
        data_list.extend([0]*(max_len-len(data_list)))
    return data_list

def remove_stop(sentences):
    '''
    去除停用词
    :param sentence:
    :return:
    '''
    remove_chars = '[·’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    new_sentences = []
    for sentence in sentences:
        # 如果字符串中包含标点符号，则将标点符号去除
        new_sentences.append(re.sub(remove_chars, '', sentence))
    return new_sentences

def chose_one(chose_list,x):
    while True:
        random_element=random.choice(chose_list)
        if random_element!=x:
            return random_element

def process_virsion(match_pool,vocab,model_type,max_len,target_list=None):
    if model_type=="edit":
        return


    if model_type=="representation":
        data=[]
        count=0
        for q_a in match_pool:
            #去除停用词
            q_a[0]=remove_stop(q_a[0])
            questions,answer=q_a[0],q_a[1]
            for question in questions:

                neg_answer=chose_one(target_list,answer)



                #对每一对q a，构造一个负样本

                data.append([question,answer,1])
                data.append([question,neg_answer,0])

        return data