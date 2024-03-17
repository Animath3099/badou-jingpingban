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
def all_cut(sentence, Dict, ans, res):#词典为Dict，文本为sentence
    length = len(sentence)
    #print(length)
    res_len = len("".join(res))
    #print(res_len)
    if res_len == 7:
        ans.append(res)
        print(ans)
        return ans
    for i in range(length):#当前的处理位置为i # i 从 0 到sentence的最后一个字的下标遍历
        for j in range(i+1, length+1):# j 遍历[i + 1, length+1]区间
            if sentence[i:j] in Dict:#取出连续区间[i, j]对应的字符串sentence[i:j] ， 如果在词典中，则认为是一个词
                pre = res.copy()#https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html
                #print(pre)
                pre.append(sentence[i:j])
                #print(pre)
                ans = all_cut(sentence[j:], Dict, ans, pre)
                #print(ans)

    return ans

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

out = all_cut(sentence, Dict, [], [])
print("输出与目标元素是否相同:", len(out) == len(target))
print("********************************************")
for i in out:
    print("输出:", i, "是否在target里:", i in target)
    print("=======================================")