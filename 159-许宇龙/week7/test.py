import csv

path = "data/val_text_cat.csv"


with open(path, 'r', newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        label = row[0]
        review = row[1]

