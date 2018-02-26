import tensorflow as tf
# tensorflow 已經提供了現成模組，可以幫您下載並讀取mnist資料
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np

# 檢察指定目錄下是否有mnist資料，若沒有的話就要下載
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# 查看mnist資料
print()
print("--------------------查看mnist資料--------------------")
print('train:{0} , validation:{1} , test:{2}'.format(mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples))
# 其中有55000筆training image，有5000筆validation image，有10000筆testing image


print()
print("--------------------查看mnist的訓練資料--------------------")
print("train images : {0} , labels : {1}".format(mnist.train.images.shape, mnist.train.labels.shape))
# 會印出(55000, 784) 和 (55000, 10)
# 代表訓練資料有55000筆，每筆資料的feature有784個點、label有10個點

print()
print("--------------------查看mnist的第0筆訓練資料--------------------")
print(mnist.train.images[0])

print()
print("--------------------查看mnist的第0筆訓練資料的label--------------------")
print(mnist.train.labels[0])

def plot_image(image):
    plt.imshow(image.reshape(28,28), cmap='binary')  # 傳入的image參數原本有784個點，要先reshpae成28x28的圖形

# 輸入參數images為多個一維陣列的圖形、labels為每個圖形對應的label (為一向量)、prediction為該圖形的預測值、idx為指定索引、num為共要印出幾個圖形
def plot_image_labels_prediction(images, labels, prediction, idx, num=10):
    if num > 25:
        num = 25
    plt.gcf().set_size_inches(12,14)  # 設定顯示圖形大小
    for i in range(num):
        plt.subplot(5,5,i+1)
        plt.imshow(images[idx + i].reshape(28,28), cmap='binary')  # 需要將784個元素的一維陣列轉換成28x28的二維陣列
        title = 'label={0}'.format(np.argmax(labels[idx + i]))
        # 若 labels[idx + i] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        # 則 np.argmax(labels[idx + i]) = 2
        if len(prediction) > 0:  # 如果有傳入預測結果
            title += ', prediction={0}'.format(prediction[idx + i])
        plt.title(title)  # 設定圖形 title
        plt.xticks([])  # x軸不顯示刻度
        plt.yticks([])  # y軸不顯示刻度
    plt.show()

#plot_image(mnist.train.images[0])
plot_image_labels_prediction(mnist.train.images, mnist.train.labels, [], 0)

# 進行深度學習網路訓練時，每次訓練時會讀取一個batch的資料來進行訓練
# tensorflow的mnist模組中已經提供mnist.train.next_batch方法可批次讀取資料
batch_images_xs, batch_images_ys = mnist.train.next_batch(batch_size=100)
print()
print("--------------------查看mnist的batch訓練資料--------------------")
print("train images : {0} , labels : {1}".format(batch_images_xs.shape, batch_images_ys.shape))
# 會印出(100, 784) 和 (100, 10)
# 代表每個batch訓練資料有100筆，每筆資料的feature有784個點、label有10個點