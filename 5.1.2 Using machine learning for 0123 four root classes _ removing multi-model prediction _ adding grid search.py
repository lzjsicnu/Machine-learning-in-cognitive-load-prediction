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
#+——————————————————————————————————————网格搜索——————————————————————————————————————————————————————————
print("进行关键特征筛选，并可视化，开始")
import datetime
time1 = datetime.datetime.now()
print(time1)

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Choose best hyperparameters by RandomizedSearchCV
param_distributions = {'max_depth': range(1, 5), 'learning_rate': np.linspace(0.1, 0.5, 5)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

model = GridSearchCV(estimator=GradientBoostingClassifier(n_estimators=2, random_state=123),
                     param_grid=param_distributions, cv=kfold)
model.fit(X_train, y_train)

sorted_index = model.best_estimator_.feature_importances_.argsort()
# plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
# .best_estimator_
plt.barh(range(X.shape[1]), model.best_estimator_.feature_importances_[sorted_index])
plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting')
plt.show()

print("交叉验证结束")
time2 = datetime.datetime.now()
print(time2)

model = model.best_estimator_
model.score(X_val, y_val)
print("网格搜索模型的正确率model.score(X_val_s, y_val)：",model.score(X_val, y_val))
pred = model.predict(x_pred)
df = pd.DataFrame()
df['Predict_ID']=pred
df.to_csv('网格搜索_调参后的梯度提升预测结果.csv',index=False)

# sorted_index = model.feature_importances_.argsort()
# #plt.barh(range(X.shape[1]), model.best_estimator_.feature_importances_[sorted_index])
# plt.barh(range(X.shape[1]), model.feature_importances_[sorted_index])
# plt.yticks(np.arange(X.shape[1]), X.columns[sorted_index])
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Gradient Boosting')
# plt.show()
# # Prediction Performance
# #prob = model.predict_proba(X_test)
# pred = model.predict(X_val)
# table = pd.crosstab(y_val, pred, rownames=['Actual'], colnames=['Predicted'])
# print("table:",table)
