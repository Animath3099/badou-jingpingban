#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut1(sentence, Dict):
    target = []
    def cut_first(sentence, target, result):
        if len(sentence) == 0:
            target.append(result)
        else:
            for i in range(1, len(sentence) + 1):
                if sentence[0:i] in Dict:
                    cut_first(sentence[i:], target, result + [sentence[0:i]])
    cut_first(sentence, target, [])
    return target
results = all_cut1(sentence, Dict)
print(results)
print(len(results))

def all_cut2(sentence, Dict):
    target = []
    if sentence in Dict:
        target.append([sentence])
    for i in range(1, len(sentence) + 1):
        if sentence[0:i] in Dict:
            for j in all_cut2(sentence[i:], Dict):
                target.append([sentence[0:i]] + j)
    return target
results = all_cut2(sentence, Dict)
print(results)
print(len(results))

def all_cut3(sentence, Dict):
    cache = {}
    def cut_with_cache(sentence, cache):
        target = []
        if sentence in Dict:
            target.append([sentence])
        for i in range(1, 4):
            word = sentence[0:i]
            other = sentence[i:]
            if word in Dict:
                if other in cache:
                    for j in cache[other]:
                        target.append([word] + j)
                else:
                    for j in cut_with_cache(other, cache):
                        target.append([word] + j)
        cache[sentence] = target
        return target
    cut_with_cache(sentence, cache)
    return cache[sentence]

results = all_cut3(sentence, Dict)
print(results)
print(len(results))

#目标输出;顺序不重要
#target = [
#    ['经常', '有意见', '分歧'],
#    ['经常', '有意见', '分', '歧'],
#    ['经常', '有', '意见', '分歧'],
#    ['经常', '有', '意见', '分', '歧'],
#    ['经常', '有', '意', '见分歧'],
#    ['经常', '有', '意', '见', '分歧'],
#    ['经常', '有', '意', '见', '分', '歧'],
#    ['经', '常', '有意见', '分歧'],
#    ['经', '常', '有意见', '分', '歧'],
#    ['经', '常', '有', '意见', '分歧'],
#    ['经', '常', '有', '意见', '分', '歧'],
#    ['经', '常', '有', '意', '见分歧'],
#    ['经', '常', '有', '意', '见', '分歧'],
#    ['经', '常', '有', '意', '见', '分', '歧']
#]
