
import os

import pandas as pd



# 获取文件夹0中的所有csv文件

folder_path = './数据/3'

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]



# 读取并合并所有csv文件

merged_data = pd.DataFrame()

for file in csv_files:

    file_path = os.path.join(folder_path, file)

    data = pd.read_excel(file_path)
    print(file_path)

    merged_data = merged_data.append(data, ignore_index=True)



# 在合并后的csv最后一列添加文件夹名称0

merged_data['label'] = '3'



# 将合并后的数据保存为新的csv文件

merged_data.to_csv('3merged_data.csv', index=False)

