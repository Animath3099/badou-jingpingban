import csv


def importcsv(data):
    with open('./output/output.csv', mode='a', newline='', encoding='utf-8') as file:
        # 初始化CSV writer对象
        writer = csv.writer(file)

        # 遍历数据列表并写入每一行
        # for row in data:
        writer.writerow(data)
    file.close()

if __name__ == '__main__':
    from week7.config import Config

    keys = Config.keys()
    importcsv(keys)