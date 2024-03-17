import collections

data = {
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

finish = []
unfinish = [[]]


sentence = "经常有意见分歧"

def all_cut(sentence):
    max_len = len(max(data, key=len))
    while len(unfinish) > 0:
        s = unfinish.pop()
        all = size_cut(sentence, max_len, s)
        if all is not None:
            for i in all:
                unfinish.append(i)
    return finish




def size_cut(sentence, max_len, unfish: list):
    all = []
    char = "".join(unfish)
    stage = len(char)
    if stage == len(sentence):
        finish.append(unfish)
        return
    for i in range(1, max_len + 1):
        s = sentence[stage:stage + i]
        last = sentence[stage:]
        if len(last) < i:
            return all
        if s in data:
            copy = unfish.copy()
            copy.append(s)
            all.append(copy)
    return all



if __name__ == "__main__":
    cuts = all_cut(sentence)
    print(cuts)
    print(len(cuts))