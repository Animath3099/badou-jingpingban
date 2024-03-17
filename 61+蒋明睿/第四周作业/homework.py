# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
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
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = []
    words = []
    index = 0
    get_word(target, words, index, sentence, Dict)

    return target


def get_word(target, words, index, sentence, Dict):
    # 获取words的copy 内存应用区分
    new_words = words.copy()
    word_size = 1
    while index + word_size <= len(sentence):
        # 根据当前词长获取当前词
        word = sentence[index: index + word_size]
        # 如果识别总长度和文本长度相符 且词属于字典 或者最后一个字 则将本次分词放入返回
        if (index + word_size == len(sentence)) and (word in Dict or word_size == 1):
            new_words.append(word)
            target.append(new_words)
        # 如果未到文本长度且当前词属于字典，则将词放入本次切割，递归调用方法开始后续词切分
        elif word in Dict:
            new_words.append(word)
            get_word(target, new_words, index + word_size, sentence, Dict)
            # 获取本长度所有切分类型后将new_words还原
            new_words = words.copy()
        word_size += 1


if __name__ == "__main__":
    for words in all_cut(sentence, Dict):
        print(words)
