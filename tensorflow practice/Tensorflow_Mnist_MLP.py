import tensorflow as tf
# tensorflow 已經提供了現成模組，可以幫您下載並讀取mnist資料
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np
import time

# 檢察指定目錄下是否有mnist資料，若沒有的話就要下載
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# 查看mnist資料
print()
print("--------------------查看mnist資料--------------------")
print("train images : {0} , labels : {1}".format(mnist.train.images.shape, mnist.train.labels.shape))
# 會印出(55000, 784) 和 (55000, 10)
# 代表訓練資料有55000筆，每筆資料的feature有784個點、label有10個點
print("validation images : {0} , labels : {1}".format(mnist.validation.images.shape, mnist.validation.labels.shape))
# 會印出(5000, 784) 和 (5000, 10)
# 代表validation資料有5000筆，每筆資料的feature有784個點、label有10個點
print("test images : {0} , labels : {1}".format(mnist.test.images.shape, mnist.test.labels.shape))
# 會印出(10000, 784) 和 (10000, 10)
# 代表測試資料有10000筆，每筆資料的feature有784個點、label有10個點

# 定義layer函數，來方便建立multilayer perceptron
def layer(output_dim, input_dim, inputs, activation=None):
    # inpus 為輸入進來的二維placeholder

    # 使用常態分布的亂數產生Weight和bias
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))

    # tensorflow需用tf.matmul()方法來進行矩陣乘法
    XWb = tf.matmul(inputs, W) + b
    if activation == None:
        y = XWb
    else:
        y = activation(XWb)
    return y

# 接下來建立輸入層x，使用tf.placeholder方法建立輸入層x，placeholder是tensorflow"計算圖"的輸入
# 後續在訓練時，會傳入數字影像資料
x = tf.placeholder("float", [None, 784])
# 第一維度設定成None，因為後續我們訓練時會傳送很多數字影像，筆數不固定，所以設定為None
# 第二維度設定成784，因為輸入的數字影像像素是784點

# 建立隱藏層h1
h1 = layer(output_dim=256, input_dim=784, inputs=x, activation=tf.nn.relu)

# 最後建立輸出層
y_predict = layer(output_dim=10, input_dim=256, inputs=h1, activation=None)

# tensorflow需自行定義訓練方式
# 開始定義訓練方式
# 首先建立訓練資料label真實值的placeholder (placeholder是tensorflow"計算圖"的輸入，後續在訓練時，會傳入數字的label真實值)
y_label = tf.placeholder("float", [None, 10])
# 第一維度設定成None，因為後續我們訓練時會傳送很多數字影像，筆數不固定，所以設定為None
# 第二維度設定成10，因為輸入的數字已經使用onehot encoding轉換成10個0或1，對應到0~9的數字

# 定義loss function
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
# tf.reduce_mean() 代表計算平均

# 定義optimizer最優化方法
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
# AdamOptimizer 的learning rate設為0.001，並使用loss_function計算loss，並依照loss來更新weight和bias，使loss最小化
# 訓練方式定義完成


# 定義評估模型的準確率
correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
# tf.equal用來判斷下列真實值(y_label)與預測值(y_predict)是否相等？若相等回傳1，否則回傳0
# tf.argmax是用來將onehot encoding轉換為數字0~9。例如 onehot encoding為[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]轉換後得到2

# 再將前一步驟的計算結果correct_prediction進行平均運算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# tf.cast的功用是將correct_prediction一維張量中的每個元素其資料形態轉為float
# 我們得到一個全部是0或1的向量，最後用tf.reduce_mean取平均取得平均值 -> 即為準確率
# 完成評估模型準確率的定義


# tensorflow需自行定義訓練過程
# 首先定義訓練參數
trainEpochs = 15  # 執行15次訓練週期
batchSize = 100  # 每一個batch有100筆資料
totalBatchs = int(mnist.train.num_examples / batchSize)  # 總共有55000筆訓練資料。每個epoch，需要執行55000 / 100 = 550個batch
loss_list = []
epoch_list = []
accuracy_list = []

startTime = time.time()  # 開始計算時間
sess = tf.Session()  # 建立tensorflow session
sess.run(tf.global_variables_initializer())  # 起始化tensorflow global 變數

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        # 進行深度學習網路訓練時，每次訓練時會讀取一個batch的資料來進行訓練
        # tensorflow的mnist模組中已經提供mnist.train.next_batch方法可批次讀取資料
        batch_x, batch_y = mnist.train.next_batch(batch_size=100)

        sess.run(optimizer, feed_dict={x:batch_x ,y_label:batch_y})

    loss, acc = sess.run((loss_function, accuracy), feed_dict={x:mnist.validation.images, y_label:mnist.validation.labels})
    epoch_list += [epoch]
    loss_list += [loss]
    accuracy_list += [acc]
    print("Train Epoch:{0:2d} Loss:{1:.9f} Accuracy:{2}".format(epoch, loss, acc))

duration = time.time() - startTime  # 計算訓練所花去的總時間
print("Train Finished takes:{0}".format(duration))

# 畫出loss誤差執行結果
plt.figure('loss')
plt.plot(epoch_list, loss_list, label='loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend()

# 畫出accuaacy準確率執行結果
plt.figure('accuracy')
plt.plot(epoch_list, accuracy_list, label='accuracy')
plt.xlabel('accuracy')
plt.ylabel('epoch')
plt.legend()

plt.show()

# 現在已經完成模型的訓練，接下來評估此模型對於測試資料的準確率
print()
print("--------------------評估模型對於測試資料的準確率--------------------")
print("accuracy : {0}".format(sess.run(accuracy, feed_dict={x:mnist.test.images, y_label:mnist.test.labels})))

# 接下來我們將使用此模型進行預測
prediction_result = sess.run(tf.argmax(y_predict, 1), feed_dict={x:mnist.test.images})
# prediction_result為一個ndarray向量，內含10000個元素，每個元素為0~9的數字

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

plot_image_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result, 0)
plt.show()