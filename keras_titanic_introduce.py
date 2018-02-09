import urllib.request
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'
filepath = './data/titanic3.xls'

if os.path.isfile(filepath) == False: # 判斷指定的目錄下是否有檔案
    # 若沒有的話就到指定的網址下載檔案
    result = urllib.request.urlretrieve(url, filepath)
    print('download:', result)

# 使用pandas提供的read_excel方法，讀取xls檔
all_df = pd.read_excel(filepath) # all_df為一個dataframe
print("-----------------------print 3 row of data from datafram-----------------------")
print(all_df[:3]) # 我們可以印出xls的前三列資料來看看
# 其中有以下欄位 pclass  survived  name  sex  age  sibsp  parch  ticket   fare  cabin  embarked  boat  body  home.dest
print()

print("-----------------------選取特定欄位後的 datafram-----------------------")
# 接下來我們只選取cols 所包含的欄位
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]
print(all_df[:3]) # 我們可以印出dataframe更新後的前三列資料來看看
print()

# 接下來要對資料進行預處理才能進行機器學習
# 因為name欄位在學習過程中不會用到，只有在prediction中才會用到，所以先暫時移除
df = all_df.drop(['name'], axis=1)

# 接下來，在age欄位中有些資料為null(不存在)，我們必須將該資料填上age欄位的平均值，否則無法進行機器學習
age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

# 接下來，在fare欄位中有些資料為null(不存在)，我們必須將該資料填上fare欄位的平均值，否則無法進行機器學習
fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)

# 原本sex欄位是文字，我們必須轉換成數字才能進行機器學習
# 令female轉成0，male轉成1
df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)

# embarked欄位有3種分類 (C、Q、S)，我們必須採用OneHot encodin
x_OneHot_df = pd.get_dummies(data=df, columns=['embarked'])

print("-----------------------轉換後的 datafram-----------------------")
print(x_OneHot_df[:3])
print()

# 接下來將dataframe轉成np.array
ndarray = x_OneHot_df.values
print("-----------------------ndarray.shape-----------------------")
print(ndarray.shape) # 印出(1309, 10)，代表有1309筆data，其中每一筆data有10個欄位，第一個欄位的資料代表第幾筆label，其他9個欄位代表該筆label的9個feature
print()

# 接下來要從ndarray中取出Label和Feature
Label = ndarray[:, 0] # Label.shape = (1309, )
Features = ndarray[:, 1:] # Feature.shape = (1309, 9)

# 在來要對ndaray的Feature欄位進行標準化
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)) # 輸入參數feature_range=(0, 1)設定標準化之後的範圍在0~1
scaledFeatures = minmax_scale.fit_transform(Features) # 對Feature的所有元素作標準化
print("-----------------------標準化後的Features-----------------------")
print(scaledFeatures[:3]) # 印出標準化後的Features的前三筆資料 (每一筆資料有9個欄位)
print()

print("-----------------------Label-----------------------")
print(Label[:3]) # 印出前三筆label (每一個label的值都代表每一個人是否存活，1代表存活，0代表死亡)
print()

# 再來要將資料分成兩部分：訓練資料及測試資料
# 將資料以隨機的方式分為訓練資料及測試資料 (約80%的資料為訓練資料，20%的資料為測試資料)
msk = np.random.rand(len(all_df)) < 0.8 # 此時 msk 可能為 np.array([True, False, False ... ])
train_df = all_df[msk] # train_df 是一個dataframe，但資料數約為1309 * 0.8筆(假設是1071) (資料數不固定，因為這是由 np.random.rand(len(all_df)) < 0.8 來決定的)
test_df = all_df[~msk] # test_df 是一個dataframe，但資料數約為1309 * 0.2筆 (假設是238) (資料數不固定，因為這是由 np.random.rand(len(all_df)) < 0.8 來決定的)
print("-----------------------每個dataframe有幾筆資料-----------------------")
print('total : {0}筆\ntrain : {1}筆\ntest : {2}筆'.format(len(all_df), len(train_df), len(test_df)) )
print()

# 最後我們可以把前面preprocess data的步驟寫成函式來表示
def PreprocessData(raw_df):
    # 因為name欄位在學習過程中不會用到，只有在prediction中才會用到，所以先暫時移除
    df = raw_df.drop(['name'], axis=1)

    # 接下來，在age欄位中有些資料為null(不存在)，我們必須將該資料填上age欄位的平均值，否則無法進行機器學習
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)

    # 接下來，在fare欄位中有些資料為null(不存在)，我們必須將該資料填上fare欄位的平均值，否則無法進行機器學習
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)

    # 原本sex欄位是文字，我們必須轉換成數字才能進行機器學習
    # 令female轉成0，male轉成1
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)

    # embarked欄位有3種分類 (C、Q、S)，我們必須採用OneHot encodin
    x_OneHot_df = pd.get_dummies(data=df, columns=['embarked'])

    # 接下來將dataframe轉成np.array
    ndarray = x_OneHot_df.values

    # 接下來要從ndarray中取出Label和Feature
    Label = ndarray[:, 0]
    Features = ndarray[:, 1:]

    # 在來要對ndaray的Feature欄位進行標準化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 輸入參數feature_range=(0, 1)設定標準化之後的範圍在0~1
    scaledFeatures = minmax_scale.fit_transform(Features)  # 對Feature的所有元素作標準化

    return (scaledFeatures, Label)

# 現在將一個dataframe丟到此函式中，即可得到 Features , Label 兩個 np.array
train_Features, train_Label = PreprocessData(train_df) # train_Features.shape = (1071, 9) , train_Label.shape = (1071, )
test_Features, test_Label = PreprocessData(test_df) # test_Feature.shape = (238, 9) , test_Label.shape = (238, )

print("-----------------------train_Features & train_Label-----------------------")
# 印出前三筆train_Features和train_Label
print(train_Features[:3])
print(train_Label[:3])







