#week3作业
from collections import defaultdict

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

def len_dict(dictionary):
    # 将字典按照长度进行划分
    ddict = defaultdict(list)
    for key,_ in dictionary.items():
        ddict[len(key)].append[key]
    return ddict
            
### 正向 ，反向 ， 正反两项
## 正向从前向后寻找最长的
def forward_max_match(text, dictionary):
    """
    使用正向最大匹配算法进行中文分词。
    
    :param text: str, 待分词的中文文本。
    :param dictionary: set, 词典，包含所有可能的词汇。
    :return: list, 分词结果列表。
    """
    out = []
    score = []
    # 函数实现部分
    max_len = max([len(key) for key in dictionary.keys()])
    i = 0
    while(i<len(text)):
        for j in range(max_len,0,-1):
            term = text[i:i+j]
            if term in dictionary.keys():
                out.append(term)
                score.append(dictionary[term])
                current_len = j
                break
            else: continue
        i += current_len
    return out, score


### 正向 ，反向 ， 正反两项
## 正向从前向后寻找最长的
def negativa_forward_max_match(text, dictionary):
    """
    使用正向最大匹配算法进行中文分词。
    
    :param text: str, 待分词的中文文本。
    :param dictionary: set, 词典，包含所有可能的词汇。
    :return: list, 分词结果列表。
    """
    out = []
    score = []
    # 函数实现部分
    max_len = max([len(key) for key in dictionary.keys()])
    i = len(text)
    while(i>0):
        for j in range(max_len,0,-1):
            term = text[i-j:i]
            if term in dictionary.keys():
                out.append(term)
                score.append(dictionary[term])
                current_len = j
                break
            else: continue
        i -= current_len
    return out[::-1], score
#实现全切分函数，输出根据字典能够切分出的所有的切分方式

def all_cut(sentence, Dict):
    #TODO
    out,score = negativa_forward_max_match(sentence, Dict)
    print(out)
    return out
a = all_cut(sentence, Dict)
#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]