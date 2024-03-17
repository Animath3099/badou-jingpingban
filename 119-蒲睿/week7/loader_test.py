import json

data_path = "文本分类练习.csv"
labels = []
with open(data_path, encoding="utf-8") as f:
    for line in f:
        row = line.strip().split(',', 1)
        print(row)
