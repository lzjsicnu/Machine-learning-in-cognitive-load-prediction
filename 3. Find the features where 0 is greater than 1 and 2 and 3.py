import pandas as pd

# 读取CSV文件
data = pd.read_csv('./1.合并后的数据0-3手动/data_dall_merged_data_包含了0123四类.csv')
df = pd.DataFrame(data)

a = []  # 为了存储o比1大的列的均，#label中均值0>1的包含了30列，1>0的包含了19列。
name1 = []
name2 = []
name3 = []
name_all = []
for feature in data.columns:
    #print("feature:",feature)
    # mean_value = data[feature].mean()
    mean_s_label_1 = df[df['label'] == 1][feature].mean()
    mean_s_label_0 = df[df['label'] == 0][feature].mean()
    mean_s_label_2 = df[df['label'] == 2][feature].mean()
    mean_s_label_3 = df[df['label'] == 3][feature].mean()

    #mean_s_label_5=(mean_s_label_1+mean_s_label_2+mean_s_label_3).mean()
    #mean_s_label_4 = df[df['label'] == 3,df['label'] == 2,df['label'] == 1][feature].mean()

    if (mean_s_label_0 > mean_s_label_1):
        name1.append(feature)
    if (mean_s_label_0 > mean_s_label_2):
        name2.append(feature)
    if (mean_s_label_0 > mean_s_label_3):
        name3.append(feature)
    if (mean_s_label_0 > mean_s_label_1) & (mean_s_label_0 > mean_s_label_2) & (mean_s_label_0 > mean_s_label_3):
        # 30:21:19，分别是0大于1,2,3,的数目
        # print(f"{feature}的均值label为0为：{mean_s_label_0}")
        # print(f"{feature}的均值label为1为：{mean_s_label_1}")
        b = mean_s_label_0 - mean_s_label_1
        # print("b=",b)
        # b=feature
        a.append(b)
        name_all.append(feature)
        # print(max(a))
#print(a)


print(len(name1))
print("name1=",name1)
print(len(name2))
print("name2=",name2)
print(len(name3))
print("name3=",name3)
print(len(name_all))
print("name_all=",name_all)


