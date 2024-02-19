#https://blog.csdn.net/zx1245773445/article/details/96992357
import pandas as pd
import pygal

inputfile = 'data_all.csv'
Raw_data= pd.read_csv(inputfile)
print("Raw_data:",Raw_data.head())
#columns = ['label','S3-D5', 'S18-D13','S15-D12', 'S3-D3','S13-D9','S6-D5','S14-D10','S19-D13','S21-D13']
#data = pd.DataFrame(Raw_data, columns=columns)

data = pd.DataFrame(Raw_data)

print("data:",data.head())
#data[data.index.duplicated()]
#data = data[~data.index.duplicated()]

# 对聚类档位，求20个簇的各项平均值
#data_mean = Raw_data.groupby('label')['S3-D5','S18-D13','S15-D12', 'S3-D3','S13-D9','S6-D5','S14-D10','S19-D13','S21-D13'].mean()
data_mean = Raw_data.groupby('label').mean()
#标准化数据
# from sklearn.preprocessing import scale
# data_scale = scale(data_mean)
# data_scale_df=pd.DataFrame(data_scale)
# print("data sacaled_df:",data_scale_df.head())
#试试归一化数据
# from sklearn import preprocessing
# minmax_scaler=preprocessing.MinMaxScaler()
# data_scale=minmax_scaler.fit_transform(data_mean)
# data_scale_df=pd.DataFrame(data_scale)
# print("data_scale_df.head():",data_scale_df.head())

import numpy as np
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData, ranges, minVals
newgroup, _, _ = noramlization(data_mean)

data_mean=newgroup
# from sklearn.preprocessing import normalize
# data_mean_normal=normalize(data_mean,axis=0,norm='max')
# print(data_mean_normal)


# 调用Radar这个类，并设置雷达图的填充，及数据范围
radar_chart = pygal.Radar(fill=True, range=(0, 1))
#radar_chart = pygal.Radar(fill=True, range=(data_mean.min, data_mean.max))
radar_chart.title = '0/1特征判定雷达图'
#各个分量名称
radar_chart.x_labels=[column for column in data_mean]
for i in range(len(data_mean)):
    #print('Hostname="%s" IP="%s" tags="%s"' % (name, value["ip"], tags))
    radar_chart.add(('类型--%s'%(i)),data_mean.loc[i])

radar_chart.render_to_file('特征判定雷达图.svg')