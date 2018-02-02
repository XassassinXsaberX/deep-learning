import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import cifar10
import matplotlib.pyplot as plt

label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# 輸入參數images為多個三維陣列的圖形、labels為每個圖形對應的label (為一數字)、prediction為該圖形的預測值、idx為指定索引、num為共要印出幾個圖形
def plot_image_labels_prediction(images, labels, prediction, idx, num=10):
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


# 載入cifar-10資料集。一開始會檢查使用者目錄下是否有cifar-10資料集(若沒有則會發一些時間來下載)
(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data() # 資料會存放於 C:\Users\user\.keras\datasets 目錄中

print("train data=",len(x_train_image)) # 其中有50000筆 train image
print("train data : ",x_train_image.shape) # 每筆 train image由 32 x 32 x 3 的三維陣列所組成 (row=32, col=32, RGB三個色版)。其中每個元素的值域為0 ~ 255
print("train label : ", y_train_label.shape) # 每筆 train label為一個圖形對應到的數字 (數字0~9分別代表 airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

print(" test data=",len(x_test_image)) # 有10000筆 test image
print(" test data : ",x_test_image.shape) # 每筆 test image由 32 x 32 的矩陣所組成。其中每個元素的值域為0 ~ 255
print(" test label : ", y_test_label.shape) # 每筆 test label為一個圖形對應到的數字 (數字0~9分別代表 airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
print("===========================")

print(x_test_image[2]) # 第2個test image為x_test_iage[2]，是(32, 32, 3)的3維陣列
print(x_test_image[2][4][4]) # 第2個test image中row=4,  col=4對應到的點。會有三個數值，分別代表Red Green Blue
print(y_test_label[2][0]) # 第2個test image對應到的label，為一個數字

plot_image_labels_prediction(x_train_image, y_train_label, [], 0, num=10) # 印出從0開始的10個image及對應的label (我們會將label數字轉成文字)
plt.show()


# 再來對四維的資料進行normalize
x_train_image_normalize = x_train_image.astype('float32') / 255
x_test_image_normalize = x_test_image.astype('float32') / 255

# 對label進行one-hot-encoding
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
print("train labet : ", y_TrainOneHot.shape)
print(" test label : ", y_TestOneHot.shape)