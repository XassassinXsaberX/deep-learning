import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# 載入mnist資料集。一開始會檢查使用者目錄下是否有mnist資料集(若沒有則會發一些時間來下載)
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data() # 資料會存放於 C:\Users\user\.keras\datasets 目錄中

# 將多筆 28 x 28 的二維資料reshape成4維資料
x_Train4D = x_train_image.reshape(60000, 28, 28, 1).astype('float32')
x_Test4D = x_test_image.reshape(10000, 28, 28, 1).astype('float32')

# 將一維的向量normalize
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

# 對label進行one-hot-encoding
y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

# 接下來建立模型
# 首先建立Sequential模型，之後只需要使用model.add()方法，將各個神經網路層加入模型即可
model = Sequential()

# 接下來要建立convolutional 層 還有 pooling 層
# 先建立convolutional 層
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', input_shape=(28, 28, 1), activation='relu'))
# filter=16 代表建立16個filer
# kernel_size 代表filter的 row column
# padding='same' 代表convolutional 層的輸出大小(row, col)不變
# input_shape=(28, 28, 1) 代表輸入資料為 28 x 28 的二維資料，且色版數目 = 1
# 最後的輸出資料其大小為 (28, 28, 16) 的資料，代表ouput資料的row=28, col=28, channel=16 (16個色版)

# 再建立"pooling 層"
model.add(MaxPooling2D(pool_size=(2, 2)))
# pool_size 代表pooling window的大小，若為(2, 2)代表 row 及 column方向的sample factor = 2
# 所以最後的輸出資料其大小為 (14, 14, 16)，代表ouput資料的row=28, col=28, channel=16 (16個色版)

# 建立第二層的 "convolutional 層" 還有 "pooling 層"
# 先建立"convolutional 層"
model.add(Conv2D(filters=36, kernel_size=(5,5), padding='same', activation='relu'))
# filter=36 代表建立36個filer
# kernel_size 代表filter的 row column
# padding='same' 代表"convolutional 層"的輸出大小(row, col)不變
# 最後的輸出資料其大小為 (14, 14, 36) 的資料，代表ouput資料的row=14, col=14, channel=36 (36個色版)

# 再建立"pooling 層"
model.add(MaxPooling2D(pool_size=(2, 2)))
# pool_size 代表pooling window的大小，若為(2, 2)代表 row 及 column方向的sample factor = 2
# 所以最後的輸出資料其大小為 (7, 7, 36)，代表ouput資料的row=7, col=7, channel=36 (16個色版)

# 最後要再加入Dropout 避免overfitting
model.add(Dropout(0.25))
# 每次在訓練迭代時，都會隨機放棄25%的神經元

# 再來建立 flatten 層
model.add(Flatten())
# 剛剛上一層的"pooling 層"輸出為(7, 7, 36)的三維資料，flatten的功用是將其變為一維的(7*7*36, ) = (1764, )向量

# 再來建立"隱藏層"
# 使用model.add()方法加入Dense神經網路層。Dense神經網路層的特色是所有上一層與下一層的神經元都完全連接
model.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
# "隱藏層"的神經元個數有1000個
# 利用normal distribution來初始化 weight 和 bias
model.add(Dropout(0.5)) # 加入dropout避免overfitting，每次在訓練迭代時，都會隨機放棄50%的神經元

# 再來加入"輸出層"到模型中
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
# 此時不需要設定input_dim，因為keras會自動依照上一層的units是1000個神經元，設定這一層的input_dim為1000個神經元

print(model.summary()) # 印出該模型的摘要

# 接下來要訓練此模型
# 我們使用compile方法對訓練模習進行設定
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 設定loss function為cross entropy error
# 設定模行訓練時weight、bias的更新方式為adam方法
# metrics是設定評估模型的方式是accuracy準確率

try:
    model.load_weights("./save_model/mnist_CNN.h5")
    print("成功載入模型，繼續訓練模型")

except:
    print("載入模型失敗，開始訓練模型")

# 接下來開始訓練模型
# 並把訓練過程存在train_history變數中
train_history = model.fit(x=x_Train4D_normalize, y=y_Train_OneHot,validation_split=0.2, epochs=10, batch_size=200, verbose=2)
# validation_split=0.2 代表keras再訓練前會將x_Train4D_normalize的 80%資料當成訓練資料、20%資料當成驗證資料(validation data)
# 所以有60000 * 0.8 = 48000筆資料當成訓練資料、60000 * 0.2 = 12000筆資料當成驗證資料
# epochs=10 代表訓練週期總數為10次epoch
# batch_size=200代表每次訓練會有200筆資料，所以一個epoch會訓練 48000 / 200 = 240 次
# verbose=2 代表會顯示訓練過程

# 從train_history變數中畫出訓練過程
def show_train_history(train_history, train, validation):
    plt.figure(train)
    plt.plot(train_history.history[train], label='train') # train_history.history[train] 存放每經過一個 epoch 訓練後，測試 train image 的準確率
    plt.plot(train_history.history[validation], label='validation') # train_history.history[validation] 存放每經過一個 epoch 訓練後，測試 validation image 的準確率
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(loc='upper left')

try:
    show_train_history(train_history,'acc','val_acc') # 畫出訓練後，train image及validation image對應到的準確率
    show_train_history(train_history,'loss','val_loss') # 畫出訓練後，train image及validation image對應到的loss function value
    plt.show()
except:
    pass

# 再來把模型儲存起來
model.save_weights("./save_model/mnist_CNN.h5")

# 目前已經完成模型的訓練，我們接下來要使用test image來測試，評估模型準確率
score = model.evaluate(x_Test4D_normalize,y_Test_OneHot)  # score[0] 為loss function值、score[1]為準確率
print()
print(('-----------------------using test image to evaluate the model-----------------------'))
print('accuracy={0}'.format(score[1]))

prediction = model.predict_classes(x_Test4D_normalize) # prediction存放每個test image的預測結果

# 輸入參數images為多個二維陣列的圖形、labels為每個圖形對應的label (為一數字)、prediction為該圖形的預測值、idx為指定索引、num為共要印出幾個圖形
def plot_image_labels_prediction(images, labels, prediction, idx, num=10):
    if num > 25:
        num = 25
    plt.gcf().set_size_inches(12,14) # 設定顯示圖形大小
    for i in range(num):
        plt.subplot(5,5,i+1)
        plt.imshow(images[idx + i], cmap='binary')
        title = 'label={0}'.format(labels[idx + i])
        if len(prediction) > 0: # 如果有傳入預測結果
            title += ', prediction={0}'.format(prediction[idx + i]) # 使用plt.imshow顯示圖形。傳入參數 image 是 28 x 28 的圖形、cmap參數設定為binary以黑白灰階顯示
        plt.title(title) # 設定圖形 title
        plt.xticks([]) # x軸不顯示刻度
        plt.yticks([]) # y軸不顯示刻度

    plt.show()

plot_image_labels_prediction(x_test_image, y_test_label, prediction, idx=340) # 顯示idx=340後的10張圖，分別標示該圖的label及透過模形的預測結果

# 接下來我們要建立一個confusion matrix，利用此matrix來看出資料的label及prediction是否不符
print()
print("-----------------------print the confusion matrix-----------------------")
print(pd.crosstab(y_test_label, prediction, colnames=['predict'], rownames=['label'])) # 印出這個confusion matrix

# 再來建立一個label (真實值) 與 prediction (預測值) 的dataframe
df = pd.DataFrame({'label':y_test_label, 'predict':prediction})

# dataframe可以方便我們快速查詢資料
print()
print('-----------------------using data frame from pandas to search-----------------------')
print(df[(df.label==5) & (df.predict==3)]) # 找出有哪幾個圖形其 label (真實值) = 5、prediction (預測值) = 3