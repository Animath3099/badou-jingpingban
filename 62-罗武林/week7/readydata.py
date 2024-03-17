import csv
import os
import random

# 划分训练集/验证集 8:2拆分数据样本
# label,review
def predata():
    list1 = list()
    list0 = list()
    with open('文本分类练习.csv', 'r', encoding='utf-8') as predata:
        reader = csv.reader(predata)
        for datarow in reader:
            if datarow[0]=="1":
               list1.append(datarow)
            elif datarow[0]=="0":
               list0.append(datarow)
        predata.close()
    if not os.path.exists("data"):
        os.makedirs("data")
    # 拆分20%验证正样本
    list_z_valid = random.sample(list1, (int)(len(list1)*0.2))
    for listtempz in list_z_valid:
        list1.remove(listtempz)
    # 拆分20%验证负样本
    list_f_valid = random.sample(list0, (int)(len(list0)*0.2))
    for listtempf in list_f_valid:
        list0.remove(listtempf)

    list_valid = list_z_valid + list_f_valid
    list_train = list1 + list0
    writerPredata("data/valid_text_classification.csv", list_valid)
    writerPredata("data/train_text_classification.csv", list_train)
    print("predata end!")

def writerPredata(filename, data):
    with open(filename, 'w',encoding='utf-8',newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
        file.close()

if __name__=="__main__":
    predata()