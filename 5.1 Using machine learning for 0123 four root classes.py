#https://blog.csdn.net/weixin_46277779/article/details/127057289?ops_request_misc=&request_id=&biz_id=102&utm_term=python%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-127057289.142^v87^control_2,239^v2^insert_chatgpt&spm=1018.2226.3001.4187
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('5.0train_data.csv')
data_pred=pd.read_csv('5.0test_data.csv')
X=data.copy()
feature=['S1-D1','S1-D3','S1-D5','S2-D1','S2-D2','S2-D4','S3-D3','S3-D5','S4-D1','S4-D3','S4-D4','S5-D2','S5-D4','S6-D5','S7-D4','S8-D4','S9-D6','S9-D7','S9-D8','S10-D6','S10-D8','S10-D9','S11-D7','S11-D8','S11-D10','S12-D8','S12-D9','S12-D10','S12-D11','S13-D9','S13-D11','S14-D10','S14-D11','S15-D12','S16-D14','S17-D14','S18-D12','S18-D13','S19-D13','S19-D14','S19-D15','S20-D14','S20-D16','S21-D12','S21-D13','S21-D15','S22-D14','S22-D15','S22-D16']
#x_train = X[feature].to_numpy() #梁智杰加入修改
X= X[feature]
x_pred=data_pred[feature]

y = data['label']
y_pred_ture=data_pred['label']

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y,random_state=0)


from sklearn.ensemble import AdaBoostClassifier
model0 = AdaBoostClassifier(n_estimators=50,random_state=77)
model0.fit(X_train, y_train)
model0.score(X_val, y_val)
print("训练过程中的测试正确率model0.score",model0.score(X_val, y_val))

# from sklearn.svm import SVC
# model_SVC = SVC(kernel="rbf", random_state=77)
# model_SVC.fit(X_train, y_train)
# model_SVC.score(X_val, y_val)
# print("训练过程中的测试正确率model_SVC.score",model_SVC.score(X_val, y_val))

#做预测
pred = model0.predict(x_pred)
# df = pd.DataFrame()
# df['label_pred']=pred
from sklearn.metrics import accuracy_score
accuracy_score(y_pred_ture, pred)

#如果加入标准化，正确率会不会上升？
#这里讲train和test分开进行了标准化，所以正确率不升反降。
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# x_pred_scaled=scaler.transform(x_pred)
# model1 = AdaBoostClassifier(n_estimators=5,random_state=77)
# model1.fit(X_train_scaled, y_train)
# model0.score(X_val_scaled, y_val)
# print("将train和test分开标准化后训练过程中的测试正确率model1.score",model0.score(X_val_scaled, y_val))

#_____________________________加入多个模型对比验证__________________________________________________________________
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# 逻辑回归
model1 = LogisticRegression(C=1e10)
# 线性判别分析
model2 = LinearDiscriminantAnalysis()
# K近邻
model3 = KNeighborsClassifier(n_neighbors=10)
# 决策树
model4 = DecisionTreeClassifier(random_state=77)
# 随机森林
model5 = RandomForestClassifier(n_estimators=1000, max_features='sqrt', random_state=10)
# 梯度提升
model6 = GradientBoostingClassifier(random_state=123)
# 极端梯度提升
#model7 = XGBClassifier(eval_metric=['logloss', 'auc', 'error'], n_estimators=1000,colsample_bytree=0.8, learning_rate=0.1, random_state=77)

# 支持向量机
model8 = SVC(kernel="rbf", random_state=77)
# 神经网络
model9 = MLPClassifier(hidden_layer_sizes=(16, 8), random_state=77, max_iter=10000)

model_list = [model1, model2, model3, model4, model5, model6,  model8, model9]
model_name = ['逻辑回归', '线性判别', 'K近邻', '决策树', '随机森林', '梯度提升',  '支持向量机', '神经网络']
df = pd.DataFrame()
for i in range(len(model_name)):
    model_C=model_list[i]
    name=model_name[i]
    model_C.fit(X_train, y_train)
    s=model_C.score(X_val, y_val)
    print(name+'方法在验证集的准确率为：'+str(s))

    pred = model_C.predict(x_pred)
    df['Predict_label']=pred
    csv_name=name+'的预测结果.csv'
    df.to_csv(csv_name,index=False)

#四分类问题，没办法画出roc曲线了
# from sklearn.metrics import plot_roc_curve
# for i in range(len(name)):
#     model_C=model_list[i]
#     name=model_name[i]
#     plot_roc_curve(model_C, X_val, y_val)
#     x = np.linspace(0, 1, 100)
#     plt.plot(x, x, 'k--', linewidth=1)
#     plt.show()
