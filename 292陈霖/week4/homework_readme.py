#week3作业

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
from itertools import combinations
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    s=sentence
    Dict_key = list(Dict.keys())
    cut_index = []
    for i in range(0, len(s)):
        # 记录所有可能的切分位置
        cut_index.extend(combinations(range(1, len(s)), i))
    # 取出每种切分情况
    output_f = []
    for i in cut_index:
        begin = 0
        output = []
        for j in i:
            output.append(s[begin:j])
            begin = j
        output.append(s[begin:])
        output_f.append(output)
    target=[]
    for i in output_f:
        if (set(i) & set(Dict))==set(i):
            target.append(i)

    return target
print(all_cut(sentence,Dict))
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