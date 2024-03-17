import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('文本分类练习.csv')

label = df['label']
review = df['review']

X_train, X_val, y_train, y_val = train_test_split(review, label, test_size=0.2, random_state=42)

# 将划分后的数据保存为新的csv文件
train_df = pd.DataFrame({'label':y_train, 'review': X_train})
val_df = pd.DataFrame({'label':y_val, 'review': X_val})


train_df.to_csv('train_text_cat.csv', index=False)
val_df.to_csv('val_text_cat.csv', index=False)

print("划分完成！")