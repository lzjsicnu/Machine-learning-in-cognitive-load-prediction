import pandas as pd

# 读取CSV文件
data = pd.read_csv('./1.合并后的数据0-3手动/0and3shoudong.csv')
df = pd.DataFrame(data)

name_0d3 = []
name_3d0 = []
for feature in data.columns:
    mean_s_label_0 = df[df['label'] == 0][feature].mean()
    print("mean_s_label_0", feature)
    print("mean_s_label_0", mean_s_label_0)
    mean_s_label_3 = df[df['label'] == 3][feature].mean()

    if (mean_s_label_0 > mean_s_label_3):
        name_0d3.append(feature)
    if (mean_s_label_3 > mean_s_label_0):
        name_3d0.append(feature)

print(len(name_0d3))
print("name3=",name_0d3)


print(len(name_3d0))
print("name0=",name_3d0)



