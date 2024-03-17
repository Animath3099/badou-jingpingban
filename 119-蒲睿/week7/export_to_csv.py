import csv

def export2csv(data):
    with open('output.csv', mode='a', newline='', encoding='utf-8') as file:
        # 初始化CSV writer对象
        writer = csv.writer(file)

        # 遍历数据列表并写入每一行
        for row in data:
            writer.writerow(row)
    file.close()

if __name__ == '__main__':
    from config import Config
    keys = Config.keys()
    export2csv(keys)