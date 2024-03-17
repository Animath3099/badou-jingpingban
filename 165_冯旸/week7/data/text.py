
import csv



def Read(num):
    data =[]
    maxlen = 0
    with open('文本分类练习.csv', 'r', encoding='utf-8') as f:
        # 跳过第一行
        next(f)
        # 遍历每一行数据
        for line in f:
            # 处理每一行数据
            row = line.strip().split(',', 1)
            data.append([row[0], row[1]])
            if len(row[1]) > maxlen:
                maxlen = len(row[1]) - 2
        print(maxlen)
    # return data[:num]+data[-num:], data[num:-num]
    creatfile('valid.csv', data[:num]+data[-num:])
    creatfile('train.csv', data[num:-num])


def creatfile(outpath, data):

    # 指定输出文件名及路径
    filename = outpath

    # 打开文件进行写入操作
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 将数据逐行写入CSV文件中
        for row in data:
            writer.writerow(row)
    file.close()


if __name__ == '__main__':

    Read(200)