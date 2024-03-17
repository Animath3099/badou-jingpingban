def full_segment(text, word_dict):
    n = len(text)

    # dp[i]表示text[:i]的所有切分方式
    dp = [[] for _ in range(n + 1)]
    dp[0] = [""]

    for i in range(1, n + 1):
        for j in range(i):
            word = text[j:i]
            # print("i:",i,"j:",j,"word:",word)
            if word in word_dict:
                for prev in dp[j]:
                    # print("prev:",prev)
                    # print("---")
                    dp[i].append(prev + " " + word)

    return dp[n]


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

text = "经常有意见分歧"
result = full_segment(text, Dict)
for r in result:
    print(r)