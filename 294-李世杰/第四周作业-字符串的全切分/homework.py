#week3作业
# 全切分

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

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict, ans, res):
    
    res_len = len("".join(res))
    if res_len == 7:
        ans.append(res)
        return ans
    length = len(sentence)
    for i in range(length):
        for j in range(i+1, length+1):
            if sentence[i:j] in Dict:
                pre = res.copy()
                pre.append(sentence[i:j])
                ans = all_cut(sentence[j:], Dict, ans, pre)
    return ans
max_len = max([len(key) for key, value in Dict.items()])          
a = all_cut(sentence, Dict,[],[])
print(a,len(a))
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