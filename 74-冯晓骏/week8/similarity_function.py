import json
import numpy as np
import jieba

'''

包含编辑距离和jaccard距离的实现

'''

#编辑距离
def editing_distance(string1, string2):
    matrix = np.zeros((len(string1) + 1, len(string2) + 1))
    for i in range(len(string1) + 1):
        matrix[i][0] = i
    for j in range(len(string2) + 1):
        matrix[0][j] = j
    print(matrix)
    for i in range(1, len(string1) + 1):
        for j in range(1, len(string2) + 1):
            if string1[i - 1] == string2[j - 1]:
                d = 0
            else:
                d = 1
            print('比较：',string1[i-1],string2[j-1])
            print('判断:',matrix[i - 1][j] + 1,matrix[i][j - 1] + 1,matrix[i - 1][j - 1] + d)
            matrix[i][j] = min(matrix[i - 1][j]+1 , matrix[i][j - 1]+1 , matrix[i - 1][j - 1] + d)
            print(matrix)
    edit_distance = matrix[len(string1)][len(string2)]
    return 1 - edit_distance / max(len(string1), len(string2))


#jaccard距离
def jaccard_distance(string1, string2):
    words1 = set(string1)
    words2 = set(string2)
    distance = len(words1 & words2) / len(words1 | words2)
    return distance


if __name__ == "__main__":
    a = "今天天气真不错"
    b = "估计明天天气更好"
    print(editing_distance(a, b))
    print(jaccard_distance(a, b))