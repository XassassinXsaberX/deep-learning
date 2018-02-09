import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# 輸入參數images為多個三維陣列的圖形、labels為每個圖形對應的label (為一數字)、prediction為該圖形的預測值、idx為指定索引、num為共要印出幾個圖形
def plot_image_labels_prediction(images, labels, prediction, idx, num=10):
    plt.figure('plot_image_labels_prediction')
    if num > 25:
        num = 25
    plt.gcf().set_size_inches(12,14) # 設定顯示圖形大小
    for i in range(num):
        plt.subplot(5,5,i+1)
        plt.imshow(images[idx + i], cmap='binary')
        title = '{0}'.format(label_dict[labels[idx + i][0]])
        if len(prediction) > 0: # 如果有傳入預測結果
            title += '=>{0}'.format(label_dict[prediction[idx + i]])
        plt.title(title) # 設定圖形 title
        plt.xticks([]) # x軸不顯示刻度
        plt.yticks([]) # y軸不顯示刻度

# 輸入參數images為多個三維陣列的圖形、labels為每個圖形對應的label (為一數字)、prediction為該圖形的預測值、predicted_prob為預測為每一種類別的機率、idx為指定索引
def show_predicted_prob(images, labels, prediction, predicted_prob, idx):
    plt.figure('show_predicted_prob')
    plt.title("label : {0} , predic : {1}".format(label_dict[labels[idx][0]], label_dict[prediction[idx]]))
    plt.imshow(images[idx], cmap='binary')
    str = ''
    for i in range(10):
        str += "{0} : {1}\n".format(label_dict[i], predicted_prob[idx][i])
    plt.xlabel(str)
    plt.xticks([])  # x軸不顯示刻度
    plt.yticks([])  # y軸不顯示刻度



# 載入cifar-10資料集。一開始會檢查使用者目錄下是否有cifar-10資料集(若沒有則會發一些時間來下載)
(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data() # 資料會存放於 C:\Users\user\.keras\datasets 目錄中

# 再來對四維的資料進行normalize
x_train_image_normalize = x_train_image.astype('float32') / 255
x_test_image_normalize = x_test_image.astype('float32') / 255

# 對label進行one-hot-encoding
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

# 接下來建立模型
# 首先建立Sequential模型，之後只需要使用model.add()方法，將各個神經網路層加入模型即可
model = Sequential()

# 接下來要建立convolutional 層 還有 pooling 層
# 先建立convolutional 層
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(32, 32, 3), activation='relu'))
# filter=32 代表建立32個filer
# kernel_size 代表filter的 row column
# padding='same' 代表convolutional 層的輸出大小(row, col)不變
# input_shape=(32, 32, 3) 代表輸入資料為 28 x 28 x 3 的三維資料 ( row=28, col=28, channel=3 (RGB3種) )
# 最後的輸出資料其大小為 (32, 32, 32) 的資料，代表ouput資料的row=28, col=28, channel=32 (32個色版)

# 還要再加入Dropout 避免overfitting
model.add(Dropout(0.25))
# 每次在訓練迭代時，都會隨機放棄25%的神經元

# 再建立"pooling 層"
model.add(MaxPooling2D(pool_size=(2, 2)))
# pool_size 代表pooling window的大小，若為(2, 2)代表 row 及 column方向的sample factor = 2
# 所以最後的輸出資料其大小為 (16, 16, 32)，代表ouput資料的row=16, col=16, channel=32 (32個色版)

# 再加入一次convolutional 層 還有 pooling 層
# 先建立convolutional 層
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# filter=64 代表建立64個filer
# kernel_size 代表filter的 row column
# padding='same' 代表convolutional 層的輸出大小(row, col)不變
# 最後的輸出資料其大小為 (16, 16, 64) 的資料，代表ouput資料的row=16, col=16, channel=64 (64個色版)

# 還要再加入Dropout 避免overfitting
model.add(Dropout(0.25))
# 每次在訓練迭代時，都會隨機放棄25%的神經元

# 再建立"pooling 層"
model.add(MaxPooling2D(pool_size=(2, 2)))
# pool_size 代表pooling window的大小，若為(2, 2)代表 row 及 column方向的sample factor = 2
# 所以最後的輸出資料其大小為 (8, 8, 64)，代表ouput資料的row=8, col=8, channel=64 (64個色版)

# 再來建立 flatten 層
model.add(Flatten())
# 剛剛上一層的"pooling 層"輸出為(8, 8, 64)的三維資料，flatten的功用是將其變為一維的(8*8*64, ) = (4096, )向量

# 還要再加入Dropout 避免overfitting
model.add(Dropout(0.25))
# 每次在訓練迭代時，都會隨機放棄25%的神經元

# 再來建立"隱藏層"
# 使用model.add()方法加入Dense神經網路層。Dense神經網路層的特色是所有上一層與下一層的神經元都完全連接
model.add(Dense(units=1024, kernel_initializer='normal', activation='relu'))
# "隱藏層"的神經元個數有1024個
# 利用normal distribution來初始化 weight 和 bias
model.add(Dropout(0.25)) # 加入dropout避免overfitting，每次在訓練迭代時，都會隨機放棄25%的神經元

# 再來加入"輸出層"到模型中
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

print(model.summary()) # 印出該模型的摘要

# 接下來要訓練此模型
# 我們使用compile方法對訓練模習進行設定
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 設定loss function為cross entropy error
# 設定模行訓練時weight、bias的更新方式為adam方法
# metrics是設定評估模型的方式是accuracy準確率

# 看看是不是已經有訓練好的模型，如果有的話直接載入即可
try:
    model.load_weights("./save_model/cifar10_CNN.h5")
    print("成功載入模型，繼續訓練模型")
except:
    print("載入模型失敗，開始訓練模型")

# 接下來開始訓練模型
# 並把訓練過程存在train_history變數中
train_history = model.fit(x=x_train_image_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=128, verbose=2)
# validation_split=0.2 代表keras再訓練前會將x_train_image_normalize的 80%資料當成訓練資料、20%資料當成驗證資料(validation data)
# 所以有50000 * 0.8 = 40000筆資料當成訓練資料、50000 * 0.2 = 10000筆資料當成驗證資料
# epochs=10 代表訓練週期總數為10次epoch
# batch_size=128代表每次訓練會有200筆資料，所以一個epoch會訓練 40000 / 128 = 375 次
# verbose=2 代表會顯示訓練過程

# 從train_history變數中畫出訓練過程
def show_train_history(train_history, train, validation):
    plt.figure(train)
    plt.plot(train_history.history[train],
             label='train')  # train_history.history[train] 存放每經過一個 epoch 訓練後，測試 train image 的準確率
    plt.plot(train_history.history[validation],
             label='validation')  # train_history.history[validation] 存放每經過一個 epoch 訓練後，測試 validation image 的準確率
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(loc='upper left')

try:
    show_train_history(train_history, 'acc', 'val_acc')  # 畫出訓練後，train image及validation image對應到的準確率
    show_train_history(train_history, 'loss', 'val_loss')  # 畫出訓練後，train image及validation image對應到的loss function value
    plt.show()
except:
    pass

# 再來把模型儲存起來
model.save_weights("./save_model/cifar10_CNN.h5")

# 目前已經完成模型的訓練，我們接下來要使用test image來測試，評估模型準確率
score = model.evaluate(x_test_image_normalize, y_TestOneHot)  # score[0] 為loss function值、score[1]為準確率
print()
print(('-----------------------using test image to evaluate the model-----------------------'))
print('accuracy={0}'.format(score[1]))

prediction = model.predict_classes(x_test_image_normalize) # prediction為ndarray，其shape為(10000, )，所以是一個一維陣列，每個元素存放每個test image的預測結果 ( input參數一定要normalize才行 )
plot_image_labels_prediction(x_test_image, y_test_label, prediction, idx=0, num=10) # 顯示idx=0後的10張圖，分別標示該圖的label及透過模形的預測結果
Predicted_Prob = model.predict(x_test_image_normalize) # Predicted_Prob存放每個test image預測成每個類別的各種機率( input參數一定要normalize才行 )
show_predicted_prob(x_test_image, y_test_label, prediction, Predicted_Prob, idx=3) # 顯示idx=0的圖，並標示該圖的label及透過模形的預測結果，及預測成其他類別所對應到的機率
plt.show()