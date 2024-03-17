import copy

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

sentence = "经常有意见分歧"


def all_cut(sentence, Dict):
    def generate_splits_recursive(input_str, start, result, results):
        if start == len(input_str) and len(''.join(result)) == len(input_str):
            results.append(copy.copy(result))

        for i in range(start, len(input_str)):
            for j in range(i + 1, len(input_str) + 1):
                word = input_str[i:j]
                if word in Dict:
                    result.append(word)
                    generate_splits_recursive(input_str, j, result, results)
                    result.pop()

    results = []
    generate_splits_recursive(sentence, 0, [], results)
    return results


results = all_cut(sentence, Dict)

for target in results:
    print(target)
