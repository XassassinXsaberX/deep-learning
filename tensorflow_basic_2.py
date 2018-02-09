import tensorflow as tf

# 建立"計算圖"

x = tf.Variable(4)  # 建立0維的tensor(就是純量)，shape=()
X = tf.Variable([0.4, 0.2, 0.4])  # 建立一維的tensor，shape=(3, )
ts_X = tf.Variable([[1., 1., 1.]])  # 建立二維的tensor，shape=(1, 3)
W = tf.Variable([[-0.5, -0.2],
                 [-0.3, 0.4],
                 [-0.5, 0.2]])  # 建立二維的tensor，shape=(3, 2)
b = tf.Variable([[0.1, 0.2]])

# tensorflow需用tf.matmul()方法來進行矩陣乘法
XW = tf.matmul(ts_X, W)  # 不行tf.matmul(X, W)
SUM = XW + b  # 至於矩陣加法可直接相加


# 接下來要執行"計算圖"
# 在執行之前必須先建立"Session"。在tensorflow中"Session"代表在用戶端和執行裝置之間建立連結
# 有了這個連結就可以將"計算圖"在裝置中執行，後續任何與裝置間的溝通，都必須透過Session，並且取得執行後的結果
with tf.Session() as sess:  # 開啟"Session"，sess就是session物件
    # 必須使用下列指令，起始化所有tensorflow global變數
    init = tf.global_variables_initializer()
    sess.run(init)

    print('x = ', sess.run(x))  # 使用sess.run顯示tensorflow變數
    print('x.shape =', x.shape)  # 印出()
    print('X = ', sess.run(x))  # 使用sess.run顯示tensorflow變數
    print('X.shape =', X.shape)  # 印出(3, )
    print('ts_X = ', sess.run(ts_X))  # 使用sess.run顯示tensorflow變數
    print('ts_X.shape =', ts_X.shape)  # 印出(1, 3)
    print('W = ', sess.run(W))  # 使用sess.run顯示tensorflow變數
    print('W.shape =', W.shape)  # 印出(3, 2)
    print('SUM =', sess.run(SUM))  # 使用sess.run顯示tensorflow變數
