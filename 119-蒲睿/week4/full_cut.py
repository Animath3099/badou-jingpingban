# 无词表递归全切分
def all_splits(sentence):
    if not sentence:
        return [[]]
    result = []
    for i in range(1, 3):
        current_word = sentence[:i]
        remaining_sentence = sentence[i:]
        remaining_split = all_splits(remaining_sentence)
        for split in remaining_split:
            result.append([current_word] + split)
    return result


# 类似all_splits, 有词表全切分
def full_cut(sentence, dict_tab, max_len):
    if not sentence:
        return [[]]
    result = []
    for i in range(1, max_len+1):
        if sentence[:i] in dict_tab:
            current_word = sentence[:i]
            remaining_sentence = sentence[i:]
            remaining_cut = full_cut(remaining_sentence, dict_tab, max_len)
            if remaining_cut:
                for cut in remaining_cut:
                    result.append([current_word] + cut)
            else:
                result.append([current_word])

    return result


Dict = {"经常": 0.1, "经": 0.05,
        "常": 0.001, "有意见": 0.1,
        "有": 0.1, "意见": 0.2,
        "意": 0.05, "见": 0.05,
        "见分歧": 0.05, "分歧": 0.2,
        "分": 0.1, "歧": 0.001}
sentence = '经常有意见分歧'
max_len = 0
for key in Dict.keys():
    max_len = max(max_len, len(key))

res = full_cut(sentence, dict_tab=Dict, max_len=max_len)
set_res = []
for e in res:
    if e not in set_res:
        set_res.append(e)

for r in set_res:
    print(r)
"""
['经', '常', '有', '意', '见', '分', '歧']
['经', '常', '有', '意', '见', '分歧']
['经', '常', '有', '意', '见分歧']
['经', '常', '有', '意见', '分', '歧']
['经', '常', '有', '意见', '分歧']
['经', '常', '有意见', '分', '歧']
['经', '常', '有意见', '分歧']
['经常', '有', '意', '见', '分', '歧']
['经常', '有', '意', '见', '分歧']
['经常', '有', '意', '见分歧']
['经常', '有', '意见', '分', '歧']
['经常', '有', '意见', '分歧']
['经常', '有意见', '分', '歧']
['经常', '有意见', '分歧']
"""
