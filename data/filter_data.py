import pandas as pd
import numpy as np
import json


train_file = 'data/train_with_size.csv'
test_file = 'data/test_with_size.csv'

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print(f'number of original classes: {len(train_df["class"].unique())}')
print(f'number of original groups: {len(train_df["group"].unique())}')
print(f'number of original train images: {len(train_df)}')
print(f'number of original test images: {len(test_df)}')

ratio = []
for i in train_df.index:
    ratio.append(train_df['width'][i]/train_df['height'][i])

train_df['ratio'] = ratio

print(f'original min train w/h ratio: {np.min(train_df["ratio"].values)}, max w/h ratio: {np.max(train_df["ratio"].values)}')

ratio = []
for i in test_df.index:
    ratio.append(test_df['width'][i]/test_df['height'][i])

test_df['ratio'] = ratio

print(f'original min test w/h ratio: {np.min(test_df["ratio"].values)}, max w/h ratio: {np.max(test_df["ratio"].values)}')

# condition 1: w/h <= threshold and h/w <= threshold
thres = 1.5
train_df1 = train_df.loc[(train_df['ratio'] >= 1/thres) & (train_df['ratio'] <= thres)]
test_df1 = test_df.loc[(test_df['ratio'] >= 1/thres) & (test_df['ratio'] <= thres)]

# condition 2: num_imgs/class >= num
num = 5
cls_count = train_df1['class'].value_counts()
cls_count = cls_count.loc[cls_count >= num]
cls_indices = cls_count.index

test_df1 = test_df1.loc[test_df1['class'].isin(cls_indices)]
tcls_count = test_df1['class'].value_counts()
tcls_count = tcls_count.loc[tcls_count >= num]
cls_indices = tcls_count.index

train_df2 = train_df1.loc[train_df1['class'].isin(cls_indices)]
test_df2 = test_df1.loc[test_df1['class'].isin(cls_indices)]

print(f'number of filtered train images: {len(train_df2)}')
print(f'number of filtered train classes: {len(train_df2["class"].unique())}')
print(f'number of filtered train groups: {len(train_df2["group"].unique())}')
print(f'number of filtered test images: {len(test_df2)}')
print(f'number of filtered test classes: {len(test_df2["class"].unique())}')

train_df2.to_csv('data/filtered_train_1.csv', columns=['name', 'class'], index=False)
test_df2.to_csv('data/filtered_test_1.csv', columns=['name', 'class'], index=False)

dic = dict([(c, i) for i, c in enumerate(cls_indices)])
json.dump(dic, open('data/cls_convert.json', 'w'))
