import tensorflow as tf
import numpy as np

# 建立"計算圖"
# 若我們希望能在執行"計算圖"階段才設定數值(一開始不設定數值)，就必須使用placeholder
X = tf.placeholder('float', [None, 3])
# 第一個參數設為float，是placeholder的資料形態
# 第二個參數設為[None, 3]，placeholder的形狀
# 其中第一個維度設為None，因為傳入的X筆數不限數量
# 第二個維度是每一筆的數字個數，每一筆有3個數字，所以設為3

# 使用常態分布的亂數產生Weight和bias
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))

# tensorflow需用tf.matmul()方法來進行矩陣乘法
XWb = tf.matmul(X, W) + b  # 不行tf.matmul(X, W)，而加法可以直接進行
y1 = tf.nn.relu(XWb)
y2 = tf.nn.sigmoid(XWb)

# 接下來要執行"計算圖"
# 在執行之前必須先建立"Session"。在tensorflow中"Session"代表在用戶端和執行裝置之間建立連結
# 有了這個連結就可以將"計算圖"在裝置中執行，後續任何與裝置間的溝通，都必須透過Session，並且取得執行後的結果
with tf.Session() as sess:  # 開啟"Session"，sess就是session物件
    # 必須使用下列指令，起始化所有tensorflow global變數
    init = tf.global_variables_initializer()
    sess.run(init)

    X_array = np.array([[0.4, 0.2, 0.4]])

    (_b, _W, _X, _y1) = sess.run((b, W, X, y1), feed_dict={X:X_array})  # 使用sess.run取得四個tensorflow變數
    # 執行"計算圖"時，placeholder X 以 feed_dict傳入 X_array

    print('b = \n{0}'.format(_b))
    print('W = \n{0}'.format(_W))
    print('_y1 = \n{0}'.format(_y1))  # 使用sess.run顯示tensorflow變數

    X_array = np.array([[0.4, 0.2, 0.4],
                        [0.3, 0.4, 0.5],
                        [0.3, -0.4, 0.5]])
    print('y1 = \n', sess.run(y1, feed_dict={X:X_array}))  # 使用sess.run顯示tensorflow變數。一定要設定feed_dict={X:X_array}，否則代表沒有傳入X