import urllib.request
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

url = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'
filepath = './data/titanic3.xls'

if os.path.isfile(filepath) == False: # 判斷指定的目錄下是否有檔案
    # 若沒有的話就到指定的網址下載檔案
    result = urllib.request.urlretrieve(url, filepath)
    print('download:', result)

# 使用pandas提供的read_excel方法，讀取xls檔
all_df = pd.read_excel(filepath) # all_df為一個dataframe

# 接下來我們只選取cols 所包含的欄位
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]

# 再來要將資料分成兩部分：訓練資料及測試資料
# 將資料以隨機的方式分為訓練資料及測試資料 (約80%的資料為訓練資料，20%的資料為測試資料)
msk = np.random.rand(len(all_df)) < 0.8 # 此時 msk 可能為 np.array([True, False, False ... ])
train_df = all_df[msk] # train_df 是一個dataframe，但資料數約為1309 * 0.8筆(假設是1071) (資料數不固定，因為這是由 np.random.rand(len(all_df)) < 0.8 來決定的)
test_df = all_df[~msk] # test_df 是一個dataframe，但資料數約為1309 * 0.2筆 (假設是238) (資料數不固定，因為這是由 np.random.rand(len(all_df)) < 0.8 來決定的)

# 我們可以把preprocess data的步驟寫成函式來表示
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

# 接下來建立模型
# 首先建立Sequential模型，之後只需要使用model.add()方法，將各個神經網路層加入模型即可
model = Sequential()

model.add(Dense(units=40, activation='relu', kernel_initializer='normal', input_dim=9)) # 將"輸入層"及"隱藏層"加到模型中
model.add(Dense(units=30, activation='relu', kernel_initializer='normal')) # 將"隱藏層"加到模型中，第二層之後不必在加上input_dim，model.add()會自動從上一層的 units 偵測出 input_dim
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='normal')) # 將"輸出層"加到模型中。輸出層只有一個神經元，1代表存活、0代表死亡

# 接下來要訓練此模型
# 我們使用compile方法對訓練模習進行設定
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 設定loss function為cross entropy error (因為只有一個output神經元，所以採用binary_crossentropy)
# 設定模行訓練時weight、bias的更新方式為adam方法
# metrics是設定評估模型的方式是accuracy準確率

# 接下來開始訓練模型
# 並把訓練過程存在train_history變數中
train_history = model.fit(x=train_Features, y=train_Label,validation_split=0.1, epochs=30, batch_size=30, verbose=2)
# validation_split=0.1 代表keras再訓練前會將train_Features的 90%資料當成訓練資料、10%資料當成驗證資料(validation data)
# 所以有1309 * 0.9 筆資料當成訓練資料、1309 * 0.1 筆資料當成驗證資料
# epochs=30 代表訓練週期總數為30次epoch
# batch_size=30代表每次訓練會有30筆資料，所以一個epoch會訓練 1309 * 0.9 / 30 次
# verbose=2 代表會顯示訓練過程

# 從train_history變數中畫出訓練過程
def show_train_history(train_history, train, validation):
    plt.figure(train)
    plt.plot(train_history.history[train],
             label='train')  # train_history.history[train] 存放每經過一個 epoch 訓練後，測試 training data 的準確率
    plt.plot(train_history.history[validation],
             label='validation')  # train_history.history[validation] 存放每經過一個 epoch 訓練後，測試 validation image 的準確率
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(loc='upper left')

try:
    show_train_history(train_history, 'acc', 'val_acc')  # 畫出訓練後，training data及validation data對應到的準確率
    show_train_history(train_history, 'loss', 'val_loss')  # 畫出訓練後，training data及validation data對應到的loss function value
    plt.show()
except:
    pass

# 目前已經完成模型的訓練，我們接下來要使用testing data來測試，評估模型準確率
score = model.evaluate(test_Features, test_Label)  # score[0] 為loss function值、score[1]為準確率
print()
print(('-----------------------using testing data to evaluate the model-----------------------'))
print('accuracy={0}'.format(score[1]))

# 接下來建立Jack和Rose的Series 資料
Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S'])
Rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])

# 建立Jack和Rose的 Dataframe 資料
JR_df = pd.DataFrame([list(Jack), list(Rose)], columns=['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])

# 將 JR_df 加到 all_df
all_df = pd.concat([all_df, JR_df])

# 因為Jack和Rose是後來才加入，所以必須再次執行預處理資料
all_Features, all_Label = PreprocessData(all_df)

# 可以開始執行預測
all_prob = model.predict(all_Features) # all_prob將存放 1309 + 2 人的存活機率 (因為最後有加入Jack Rose，所以會加兩人)

# 最後將all_prob與all_df整合
all_df.insert(len(all_df.columns), 'prob', all_prob)
print(all_df[-2:])


