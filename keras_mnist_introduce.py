import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 畫出 28 x 28 的 image
def plot_image(image):
    # image 為 28 x 28 且每個元素值域為0 ~ 255的矩陣
    plt.gcf().set_size_inches(2,2)  # 設定顯示圖形大小
    plt.imshow(image, cmap='binary') # 使用plt.imshow顯示圖形。傳入參數 image 是 28 x 28 的圖形、cmap參數設定為binary以黑白灰階顯示
    plt.show()

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
            title += ', prediction={0}'.format(prediction[idx + i])
        plt.title(title) # 設定圖形 title
        plt.xticks([]) # x軸不顯示刻度
        plt.yticks([]) # y軸不顯示刻度

    plt.show()

# 載入mnist資料集。一開始會檢查使用者目錄下是否有mnist資料集(若沒有則會發一些時間來下載)
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data() # 資料會存放於 C:\Users\user\.keras\datasets 目錄中

print("train data=",len(x_train_image)) # 其中有60000筆 train image
print("train data : ",x_train_image.shape) # 每筆 train image由 28 x 28 的矩陣所組成。其中每個元素的值域為0 ~ 255
print("train label : ", y_train_label.shape) # 每筆 train label為一個圖形對應到的數字

print(" test data=",len(x_test_image)) # 有10000筆 test image
print(" test data : ",x_test_image.shape) # 每筆 test image由 28 x 28 的矩陣所組成。其中每個元素的值域為0 ~ 255
print(" test label : ", y_test_label.shape) # 每筆 test label為一個圖形對應到的數字
print("===========================")

#plot_image(x_train_image[0])  # 顯示第0筆 train image
#print(y_train_label[0]) # 印出第0筆 train image對應到的label (為一個數字)

plot_image_labels_prediction(x_test_image, y_test_label, [], 0, 10) # 印出從0開始的10個image及對應的label

# 將 28 x 28 的二維資料reshape成一維的向量
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
print("train data : ", x_Train.shape)
print(" test data : ", x_Test.shape)

# 將一維的向量normalize
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

# 對label進行one-hot-encoding
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
print("train labet : ", y_TrainOneHot.shape)
print(" test label : ", y_TestOneHot.shape)
