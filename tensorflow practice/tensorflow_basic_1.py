import tensorflow as tf

# 建立"計算圖"
ts_c = tf.constant(2, name='ts_c')  # 建立tensorflow常數，並設定常數值為2、常數名稱為name = "ts_c"，此名稱會顯示在計算圖上
ts_x = tf.Variable(ts_c + 5, name='ts_x')  # 建立tensorflow變數，並設定變數值為ts_c + 5、變數名稱為name = "ts_x"，此名稱會顯示在計算圖上

# 若我們希望能在執行"計算圖"階段才設定數值，就必須使用placeholder
width = tf.placeholder("int32", name='width')  # 變數名稱為name = "weight"，此名稱會顯示在計算圖上
height = tf.placeholder("int32", name='height')  # 變數名稱為name = "height"，此名稱會顯示在計算圖上
area = tf.multiply(width, height, name='area')  # 使用tf.multiply() 將 width 與 height 相乘變，變數名稱為name = "area"，此名稱會顯示在計算圖上
# https://www.tensorflow.org/api_guides/python/math_ops 這裡有更多tensorflow的數學運算

# 接下來要執行"計算圖"
# 在執行之前必須先建立"Session"。在tensorflow中"Session"代表在用戶端和執行裝置之間建立連結
# 有了這個連結就可以將"計算圖"在裝置中執行，後續任何與裝置間的溝通，都必須透過Session，並且取得執行後的結果
with tf.Session() as sess:  # 開啟"Session"，sess就是session物件
    # 必須使用下列指令，起始化所有tensorflow global變數
    init = tf.global_variables_initializer()
    sess.run(init)

    print('ts_c=', sess.run(ts_c))  # 使用sess.run顯示tensorflow常數
    print('ts_x=', sess.run(ts_x))  # 使用sess.run顯示tensorflow變數
    print('area=', sess.run(area, feed_dict={width:6, height:8}))

    # 接下來，下列程式碼將要顯示在TensorBoard的資料，寫入到log檔
    tf.summary.merge_all()  # 將所有要顯示在TensorBoard的資料整合
    train_writer = tf.summary.FileWriter("./log/area", sess.graph)