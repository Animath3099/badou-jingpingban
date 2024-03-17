# week3作业
# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {
        "经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1
}

# 待切分文本
string = "经常有意见分歧"

def gen_prefix_dic():
    dic = {}
    for word in Dict.keys():
        for i in range(1, len(word)):
            if word[:i] not in dic:  # 不能用前缀覆盖词
                dic[word[:i]] = 0  # 前缀
        dic[word] = 1  # 词
    return dic



def circle(start_index, words):
    # 2中情况， 多个1 或者没有
    find_word_arr = []
    end_index = start_index + 1
    window = string[start_index:end_index]

    while start_index < len(string):
        # if end_index == len(string):
        #     words.append(window)
        #     print(words)
        #     words.pop()
        #     break

        # 窗口没有在词典里出现
        if window not in prefix_dict or end_index > len(string):
            # 开始处理arr
            for w in find_word_arr:
                words.append(w)
                # if start_index + len(window) == len(string):
                #     print(words)
                #     break
                #print(words)
                start_index += len(w)
                if start_index < len(string):
                    circle(start_index, words)
                else:
                    print(words)
                words.pop()
                start_index -= len(w)
            break
            #start_index -= len(window)
        # 窗口是个词
        elif prefix_dict[window] == 1:
            find_word_arr.append(window)
            # 往后再看
            # if end_index < len(string):
            end_index += 1
            window = string[start_index:end_index]

        # 窗口是一个前缀
        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index:end_index]


def cut_method():

    words = []
    start_index = 0
    circle(start_index, words)

    print(words)





if __name__ == '__main__':
    prefix_dict = gen_prefix_dic()
    print(prefix_dict)
    cut_method()


# 运行结果：
# {'经': 1, '经常': 1, '有': 1, '常': 1, '有意': 0, '有意见': 1, '歧': 1, '意': 1, '意见': 1, '分': 1, '分歧': 1, '见': 1, '见分': 0, '见分歧': 1}
# ['经', '常', '有', '意', '见', '分', '歧']
# ['经', '常', '有', '意', '见', '分歧']
# ['经', '常', '有', '意', '见分歧']
# ['经', '常', '有', '意见', '分', '歧']
# ['经', '常', '有', '意见', '分歧']
# ['经', '常', '有意见', '分', '歧']
# ['经', '常', '有意见', '分歧']
# ['经常', '有', '意', '见', '分', '歧']
# ['经常', '有', '意', '见', '分歧']
# ['经常', '有', '意', '见分歧']
# ['经常', '有', '意见', '分', '歧']
# ['经常', '有', '意见', '分歧']
# ['经常', '有意见', '分', '歧']
# ['经常', '有意见', '分歧']