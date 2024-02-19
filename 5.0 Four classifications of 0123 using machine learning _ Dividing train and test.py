import pandas as pd
import numpy as np
file_path = './1.合并后的数据0-3手动/data_dall_merged_data_包含了0123四类.csv'  # 请替换为您的CSV文件路径
df = pd.read_csv(file_path)
#df['ID'] = range(len(df))
# 将添加了ID列的数据保存到新的csv文件中
#df.to_csv('3.下采样后的数据.csv', index=False)

np.random.seed(42)  # 设置随机种子以确保每次运行结果相同
shuffled_index = np.random.permutation(df.index)
train_df = df.loc[shuffled_index[:int(len(df)*0.8)], :]  # 取前0.8比例的数据作为训练集
test_df = df.loc[shuffled_index[int(len(df)*0.8):], :]  # 取剩余数据作为测试集


train_df.to_csv('5.0train_data.csv', index=False)  # 将训练集保存到train_data.csv文件中，不包括行索引
test_df.to_csv('5.0test_data.csv', index=False)  # 将测试集保存到test_data.csv文件中，不包括行索引