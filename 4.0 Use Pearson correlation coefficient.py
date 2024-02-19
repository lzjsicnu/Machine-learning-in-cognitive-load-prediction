# data analysis and Wangling's data
import pandas as pd
import numpy as np
import random as rnd
from sklearn.model_selection import train_test_split,cross_val_score

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[7]:


#使用pandas管理数据
#data_df = pd.read_csv(r'3.下采样后train_data_删除ID.csv')
data_df = pd.read_csv(r'4.下采样后test_data_删除ID_用a和b做标签重新画相关性图_把1和0返一下.csv')

#Info()方法探索空值
print(data_df.info())
data_df.head()

data_df.describe()
#data = data_df.values
# rownames = data_df.index
# colnames = data_df.columns

data_df['label'].value_counts().to_dict()
cols=data_df.columns.values.tolist()

df1=data_df.copy()
#找出与标签y相关性最强的前10项特征
corrmat=df1.corr()
k=10
cols_corr=corrmat.nlargest(k,"label")['label'].index
cm=np.corrcoef(df1[cols_corr].values.T)
sns.set(font_scale=1.25)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},
               yticklabels=cols_corr.values,xticklabels=cols_corr.values)
plt.show()

x=df1.loc[:,[col for col in cols if col!='label']]
print(x.shape)

y=df1.loc[:,['label']]
print((y.shape))