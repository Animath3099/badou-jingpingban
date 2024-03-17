#week3作业

import copy

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
def all_cut(sentence, Dict):
    ## 词典中所有的词
    dictKeys = Dict.keys()
    ## 词典中最长的词
    maxLen = max(len(s) for s in dictKeys)
    ## 所有切分结果
    all_cut = []
    ## 切分函数
    cut_str_to_arr(sentence, maxLen, dictKeys, all_cut)
    ## 结果处理
    ## 使用集合去重，将每个子数组转换为元组
    unique_data = set(tuple(sub_array) for sub_array in all_cut)
    ## 将去重后的结果转换回列表形式
    all_cut = [list(sub_tuple) for sub_tuple in unique_data]
    print(all_cut)
    print(len(all_cut))
    return all_cut

# sentence 字符串
# max_len 词典中最长的词
# dictKeys 词典
# all_cut 所有切分结果
# position 已经切到的位置
# front_arr 已经切了的内容的切分结果
def cut_str_to_arr(sentence, max_len, dictKeys, all_cut,  position = 0, front_arr = []):
    keys = []
    end_len = min(max_len, len(sentence) - position);
    word = sentence[position:position + end_len]
    for i in range(1, end_len + 1):
        if word[:i] in dictKeys:
            keys.append(word[:i])

    for key in keys:
        words = copy.deepcopy(front_arr)
        string = sentence[position:]
        while string != '':
            lens = min(max_len, len(string))
            word1 = string[:lens]
            while word1 not in dictKeys:
                if len(word1) == 1:
                    break
                word1 = word1[:len(word1) - 1]
            words.append(word1)
            string = string[len(word1):]
        all_cut.append(words)

        cur_front_arr = copy.deepcopy(front_arr)
        cur_front_arr.append(key)
        if(end_len > 1):
            cut_str_to_arr(sentence, max_len, dictKeys, all_cut, position + len(key), cur_front_arr)

def main():
    all_cut(sentence, Dict)
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

if __name__ == "__main__":
    main()