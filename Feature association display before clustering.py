import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
# 忽略警告
warnings.filterwarnings("ignore")
#图片在notebook内展示
#%matplotlib inline
## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

# 利用pandas读入数据
Raw_data=pd.read_csv("data_all.csv")
# 查看前5行
columns = ['S3-D5', 'S18-D13','S15-D12', 'S3-D3','S13-D9','S6-D5','S14-D10','S19-D13','S21-D13','S21-D13']
df = pd.DataFrame(Raw_data, columns=columns)
df.head()
#求各产品之间的相关系数，并绘制相关性的热力图：
plt.figure(figsize=(4,4),dpi=150)
corr=df.corr() #求各产品之间的相关系数
sns.heatmap(corr,cmap="Reds",annot=True) # 并绘制相关性的热力图
plt.show()


#数据标准化，消除数据值的差异对聚类的影响
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# print("df:",df)
# df_scaled = scaler.fit_transform(df)
# print("df_scaled:",df_scaled)


