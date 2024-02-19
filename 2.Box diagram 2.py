#https://blog.csdn.net/xu624735206/article/details/120449285
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv('./1.合并后的数据0-3手动/data_dall_merged_data_包含了0123四类.csv')


# 根据“S6”特征列绘制垂直箱线图

plt.figure(figsize=(10, 60))
y='S6-D5'
sns.boxplot(x='label', y=y, data=data)

#plt.title('Vertical Boxplot of S3-D5')
plt.title('Boxplot of 'f"{y}")

plt.show()

