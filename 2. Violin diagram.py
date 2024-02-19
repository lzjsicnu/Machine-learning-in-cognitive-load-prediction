
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# 读取CSV文件
#data = pd.read_csv('./1.合并后的数据0-3手动/data_dall_merged_data.csv')
data = pd.read_csv('data_dall_merged_data_包含了0123四类手动.csv')



# 根据“S6”特征列绘制垂直小提琴图

plt.figure(figsize=(10, 6))
y='S5-D4'
sns.violinplot(x='label',y=y, data=data)
plt.title('Violin of 'f"{y}")

plt.show()

