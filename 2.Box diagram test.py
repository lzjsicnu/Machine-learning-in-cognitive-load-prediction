import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#data = pd.read_csv('./1.合并后的数据0-3手动/data_dall_merged_data_包含了0123四类.csv')
#data = pd.read_csv('./1.合并后的数据0-3手动/data_all只包含0和1.csv')


if __name__ == '__main__':

    # a_pd = pd.DataFrame({'A': [1, 2, 3], "B": [0, 6, 5]})
    # b_pd = pd.DataFrame({'A': [1, 2, 2], "B": [4, 6, 5]})
    # c_pd = pd.DataFrame({'A': [7, 8, 9], "B": [5, 6, 7]})

    #S5D4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    a_pd = pd.DataFrame({'O': [4, 6, 8.1]})
    b_pd = pd.DataFrame({'1': [2, 3.5, 5.5]})
    c_pd = pd.DataFrame({'2': [1, 3.8, 8.5]})
    d_pd = pd.DataFrame({'3': [2.8, 3.6, 5.3]})

    new_pd = pd.concat([a_pd,b_pd,c_pd,d_pd], keys=('Englist', 'Argbar','Geo','Scale'))
    print(new_pd)
    "stack: 行转列，由DataFrame转化为Series"
    new_pd = new_pd.stack()
    print(new_pd)
    "加入列名"
    new_pd = new_pd.rename_axis(index=['Feature', 'nan', 'Data Size'])
    print(new_pd)
    "由Series转化为DataFrame"
    new_pd = new_pd.reset_index(level=[0, 2], name='value')
    print(new_pd)

    ax = sns.boxplot(data=new_pd, x='Data Size', hue='Feature', y='value')
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    plt.show()




  # S3D5~~~~~~~~~~~~~~~~~~~~~~~~~~~
    a_pd = pd.DataFrame({'O': [1.2, 2.8, 7.1]})
    b_pd = pd.DataFrame({'1': [2, 4.5, 8.5]})
    c_pd = pd.DataFrame({'2': [2.2, 4.68, 8.6]})
    d_pd = pd.DataFrame({'3': [2.1, 4.6, 7.3]})
    new_pd = pd.concat([a_pd, b_pd, c_pd, d_pd], keys=('Englist', 'Argbar','Geo','Scale'))
    print(new_pd)
    "stack: 行转列，由DataFrame转化为Series"
    new_pd = new_pd.stack()
    print(new_pd)
    "加入列名"
    new_pd = new_pd.rename_axis(index=['Feature', 'nan', 'Data Size'])
    print(new_pd)
    "由Series转化为DataFrame"
    new_pd = new_pd.reset_index(level=[0, 2], name='value')
    print(new_pd)

    ax = sns.boxplot(data=new_pd, x='Data Size', hue='Feature', y='value')
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    plt.show()



    # S6D5~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    a_pd = pd.DataFrame({'O': [1.2, 2.8, 7.1]})
    b_pd = pd.DataFrame({'1': [2, 4.8, 8.3]})
    c_pd = pd.DataFrame({'2': [3.2, 4.71, 8.4]})
    d_pd = pd.DataFrame({'3': [1.23, 4.69, 8.31]})
    new_pd = pd.concat([a_pd, b_pd, c_pd, d_pd], keys=('Englist', 'Argbar','Geo','Scale'))
    print(new_pd)
    "stack: 行转列，由DataFrame转化为Series"
    new_pd = new_pd.stack()
    print(new_pd)
    "加入列名"
    new_pd = new_pd.rename_axis(index=['Feature', 'nan', 'Data Size'])
    print(new_pd)
    "由Series转化为DataFrame"
    new_pd = new_pd.reset_index(level=[0, 2], name='value')
    print(new_pd)
    ax = sns.boxplot(data=new_pd, x='Data Size', hue='Feature', y='value')
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    plt.show()

    # S10D6~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    a_pd = pd.DataFrame({'O': [2.8, 3.8, 6.1]})
    b_pd = pd.DataFrame({'1': [2.76, 3.89, 7.3]})
    c_pd = pd.DataFrame({'2': [1.81, 3.81, 8.4]})
    d_pd = pd.DataFrame({'3': [2.90, 3.90, 8.31]})
    new_pd = pd.concat([a_pd, b_pd, c_pd, d_pd], keys=('Englist', 'Argbar','Geo','Scale'))
    print(new_pd)
    "stack: 行转列，由DataFrame转化为Series"
    new_pd = new_pd.stack()
    print(new_pd)
    "加入列名"
    new_pd = new_pd.rename_axis(index=['Feature', 'nan', 'Data Size'])
    print(new_pd)
    "由Series转化为DataFrame"
    new_pd = new_pd.reset_index(level=[0, 2], name='value')
    print(new_pd)
    ax = sns.boxplot(data=new_pd, x='Data Size', hue='Feature', y='value')
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    plt.show()

