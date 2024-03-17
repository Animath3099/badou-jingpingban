import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
def evalute_model(model,data,algorithm,class_questions=None,dataset=None):
    right=0
    wrong=0
    ACC=0
    start=time.time()
    print("开始模型评估测试")
    if algorithm=='edit':
        print("基于编辑距离相似度算法评估")
        for sample in data:
            question,target=sample[0],sample[1]
            answer,score=model.query(question)
            if answer==target:
                right+=1
            else:
                wrong+=1
    elif algorithm=='represent':
        assert class_questions is not None
        print("表示型架构的算法评估")
        model.eval()
        question_idx_to_class={}
        questions_idx=[]

        for class_,questions in class_questions.items():
            for question in questions:
                question_idx_to_class[len(questions_idx)]=class_
                questions_idx.append(question)
        with torch.no_grad():
            #竖着拼起来拼成 batch_size,max_len的形式,并将他们向量化
            #计算向量的余弦相似度
            question_matrix=torch.stack(questions_idx,dim=0)
            vector_question_matrix=model(question_matrix)
            vector_question_matrix=torch.nn.functional.normalize(vector_question_matrix, dim=-1)

            for idx,sample in enumerate(data):
                user_question,target=sample[0],sample[1]
                user_question=model(user_question)
                res=torch.mm(user_question,vector_question_matrix.T)
                hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
                hit_class = question_idx_to_class[hit_index]

                if hit_class==target:
                    right+=1
                else:
                    wrong+=1
    elif algorithm=="interaction":
        assert dataset is not None
        print("交互型架构的算法评估")
        idx_to_class={}#一个从问题下标到问题所属类别的映射
        questions_list=[]#问题列表

        #data:valid_data
        for idx,sample in enumerate(data):
            concat_list = []
            user_question,target=sample[0],sample[1]
            score_list=[]
            for class_,questions in class_questions.items():
                for question in questions:
                    idx_to_class[len(questions_list)]=class_
                    questions_list.append(question)
                    concat_question=dataset.sentence_encoder(user_question,question)
                    #将各个类别各个问题和用户问题进行拼接
                    concat_list.append(concat_question)
            concat_list=torch.LongTensor(concat_list)
            sim=model(concat_list).squeeze()
            sim=sim.cpu().detach().numpy()
            idx=idx_to_class[np.argmax(sim)]
            if idx==target.item():
                right+=1
            else:
                wrong+=1
    ACC=right/(right+wrong)
    end=time.time()
    print("测试数据运行时间:",round(end-start,2),'秒')
    print("测试数据量:{}条,准确率:{}%".format(len(data),round(ACC*100,2)))