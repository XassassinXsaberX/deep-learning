import tensorflow as tf
import numpy as np

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

# 建立"計算圖"
# 若我們希望能在執行"計算圖"階段才設定數值(一開始不設定數值)，就必須使用placeholder
X = tf.placeholder('float', [None, 4])  # X為輸入層
# 第一個參數設為float，是placeholder的資料形態
# 第二個參數設為[None, 4]，placeholder的形狀
# 其中第一個維度設為None，因為傳入的X筆數不限數量
# 第二個維度是每一筆的數字個數，每一筆有4個數字，所以設為4

h = layer(output_dim=3, input_dim=4, inputs=X, activation=tf.nn.relu)  # h為隱藏層
y = layer(output_dim=2, input_dim=3, inputs=h)  # y為輸出層


# 接下來要執行"計算圖"
# 在執行之前必須先建立"Session"。在tensorflow中"Session"代表在用戶端和執行裝置之間建立連結
# 有了這個連結就可以將"計算圖"在裝置中執行，後續任何與裝置間的溝通，都必須透過Session，並且取得執行後的結果
with tf.Session() as sess:  # 開啟"Session"，sess就是session物件
    # 必須使用下列指令，起始化所有tensorflow global變數
    init = tf.global_variables_initializer()
    sess.run(init)

    X_array = np.array([[0.4, 0.2, 0.4, 0.5]])

    (layer_X, layer_h, layer_y) = sess.run((X, h, y), feed_dict={X:X_array})  # 使用sess.run取得三個tensorflow變數
    # 執行"計算圖"時，placeholder X 以 feed_dict傳入 X_array

    print('layer_X = \n{0}'.format(layer_X))
    print('layer_h = \n{0}'.format(layer_h))
    print('layer_y = \n{0}'.format(layer_y))

