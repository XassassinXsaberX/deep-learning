import urllib.request
import os, re
import tarfile
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
filepath = './data/aclImdb_v1.tar.gz'

if os.path.isfile(filepath) == False: # 判斷指定的目錄下是否有壓縮檔
    # 若沒有的話就到指定的網址下載該壓縮檔
    result = urllib.request.urlretrieve(url, filepath)
    print('download:', result)

if os.path.exists('./data/aclImdb') == False: # 判斷指定目錄下是否有aclImdb目錄
    # 若沒有此目錄的話
    tfile = tarfile.open(filepath, 'r:gz') # 開啟指定壓縮檔
    result = tfile.extractall('./data') # 解壓縮檔案到./data

# 下載完，且解壓縮後可以開始讀取IMDb的資料了
# 首先我們定義一個函式，使用regular expression 移除HTML tag
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>') # 建立regular expression變數為re_tag
    return re_tag.sub(' ', text) # 使用re_tag 將 text 文字中，符合regular expression條件的字，替換成空字串

# 再來建立read_files函數讀取IMDb檔案目錄
def read_files(filetype):
    path = "./data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path): # os.listdir(positive_path) 會列出 positive_path 目錄下有哪些目錄及檔案
        file_list += [positive_path + f] # 將 positive_path 目錄下的目錄及檔案字串放到 file_list 中

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path): # os.listdir(negative_path) 會列出 negative_path 目錄下有哪些目錄及檔案
        file_list += [negative_path + f] # 將 negative_path 目錄下的目錄及檔案字串放到 file_list 中

    print("read {0} files : {1}".format(filetype, len(file_list)))

    all_labels = ([1]*12500 + [0]*12500) # 前12500筆資料是positive、後12500筆資料是negative
    all_texts = []

    for fi in file_list:
        with open(fi, encoding = 'utf-8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            # 使用file_input.readlines() 一次讀取檔案的所有字串，此時會得到一個list物件，其中的每一個list中的元素為檔案中每一行的字串
            # 用join將list中多個元素(即為字串) 連接起來，此時我們得到一字串
            # 最後使用rm_tags函數，將該字串的HTML tag移除

    return all_labels, all_texts

# 現在用read_files函數讀取資料
train_label, train_text = read_files("train") # train_label為一個有25000個元素的list，每個list為0或1的整數、train_text為一個有25000個元素的list，每個元素為string
test_label, test_text = read_files("test") # test_label為一個有25000個元素的list，每個list為0或1的整數、test_text為一個有25000個元素的list，每個元素為string

# 接下來要建立"字典"
token = Tokenizer(num_words=2000) # 建立一個有2000字的字典
token.fit_on_texts(train_text)
# 讀取所有的訓練資料影評，依照每一個英文字，在影評中出現的次數進行排序
# 排序的前2000名的英文字，會列入字典中

# 接下來將影評文字轉換成數字list
x_train_seq = token.texts_to_sequences(train_text) # 為一個有25000個元素的list，每個元素也是一個list，該list的元素則存放著轉換後的數字
x_test_seq = token.texts_to_sequences(test_text) # 為一個有25000個元素的list，每個元素也是一個list，該list的元素則存放著轉換後的數字

# 接下來使用sequence.pad_sequences() 方法將數字list截長補短，使所有數字list的長度都為100
# 若數字list的長度=126，就將前面的26個數字截去
# 若數字list的長度為59，就在前面補上41個0
x_train = sequence.pad_sequences(x_train_seq, maxlen=100) # 為一個有25000個元素的ndarray，每個元素是一個list，該list的元素則存放著轉換後的數字 (現在強制變為100個)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100) # 為一個有25000個元素的ndarray，每個元素是一個list，該list的元素則存放著轉換後的數字 (現在強制變為100個)

# 接下來建立模型
# 首先建立Sequential模型，之後只需要使用model.add()方法，將各個神經網路層加入模型即可
model = Sequential()

# 接下來要加入"Embedding 層"到模型中
# keras提供"Embedding 層" 可以將 "數字list" 轉成 "向量list"
# 這是word embedding的技術。剛剛將一個英文字轉成對應到的一個數字，現在將這個數字轉成一個向量
model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
# ouput_dim=32代表我們希望將"數字list"轉換成32維的向量
# imput_dim=2000 因為我們之前建立的字典含有2000個字
# input_length=100 因為每一個"數字list"中有100個數字

model.add(Dropout(0.2)) # 加入dropout避免overfitting，每次在訓練迭代時，都會隨機放棄20%的神經元

# 將"數字list"轉換成"向量list"後，我們可以開始建立 multilayer perceptron 進行深度學習
# 首先要用"flatten 層"將input轉換為一維向量
# 因為原本的"數字list"有100個數字，但轉換成"向量list後"，每一個數字轉成32維的向量，所以"向量list"為一個二維向量
model.add(Flatten())
# 經過"flatten 層"後，input的神經元變為100 * 32 = 3200個

model.add(Dense(units=256, activation='relu', kernel_initializer='normal')) # 將"隱藏層"加到模型中
model.add(Dropout(0.35)) # 加入dropout避免overfitting，每次在訓練迭代時，都會隨機放棄35%的神經元
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='normal')) # 將"輸出層"加到模型中。輸出層只有一個神經元，1代表正面評價、0代表負面評價

print(model.summary()) # 印出該模型的摘要

# 接下來要訓練此模型
# 我們使用compile方法對訓練模習進行設定
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 設定loss function為cross entropy error (因為只有一個output神經元，所以採用binary_crossentropy)
# 設定模行訓練時weight、bias的更新方式為adam方法
# metrics是設定評估模型的方式是accuracy準確率

# 看看是不是已經有訓練好的模型，如果有的話直接載入即可
try:
    model.load_weights("./save_model/IMDB_MLP.h5")
    print("成功載入模型，繼續訓練模型")
except:
    print("載入模型失敗，開始訓練模型")

# 接下來開始訓練模型
# 並把訓練過程存在train_history變數中
#train_history = model.fit(x=x_train, y=np.array(train_label), validation_split=0.2, epochs=10, batch_size=100, verbose=2)
# validation_split=0.2 代表keras再訓練前會將train_Features的 90%資料當成訓練資料、10%資料當成驗證資料(validation data)
# 所以有25000 * 0.9 筆資料當成訓練資料、25000 * 0.1 筆資料當成驗證資料
# epochs=10 代表訓練週期總數為10次epoch
# batch_size=100 代表每次訓練會有100筆資料，所以一個epoch會訓練 25000 * 0.9 / 100 次
# verbose=2 代表會顯示訓練過程

# 從train_history變數中畫出訓練過程
def show_train_history(train_history, train, validation):
    plt.figure(train)
    plt.plot(train_history.history[train], label='train')  # train_history.history[train] 存放每經過一個 epoch 訓練後，測試 training data 的準確率
    plt.plot(train_history.history[validation], label='validation')  # train_history.history[validation] 存放每經過一個 epoch 訓練後，測試 validation image 的準確率
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(loc='upper left')

try:
    show_train_history(train_history, 'acc', 'val_acc')  # 畫出訓練後，training data及validation data對應到的準確率
    show_train_history(train_history, 'loss', 'val_loss')  # 畫出訓練後，training data及validation data對應到的loss function value
    plt.show()
except:
    pass

# 再來把模型儲存起來
model.save_weights("./save_model/IMDB_MLP.h5")

# 目前已經完成模型的訓練，我們接下來要使用testing data來測試，評估模型準確率
score = model.evaluate(x_test, np.array(test_label))  # score[0] 為loss function值、score[1]為準確率
print()
print(('-----------------------using testing data to evaluate the model-----------------------'))
print('accuracy={0}'.format(score[1]))

# 再來使用此已訓練好的模型進行預測
prediction = model.predict_classes(x_test) # prediction存放每個testing data的預測結果
# prediction 為一個ndarray，其shape為(25000, 1)的二維陣列

prediction_class = prediction.reshape(-1) # 將prediction_class變為(25000, )的一維陣列

# 建立一個函數，可以顯示某篇testing文章的內容，及評價
SentimentDict = {1:"正面的", 0:"負面的"}
def display_test_Sentiment(i):
    print(test_text[i])
    print("label真實值：{0} , 預測結果：{1}".format(SentimentDict[test_label[i]], SentimentDict[prediction_class[i]]))

print()
print("-----------------------顯示第三篇testin文章的內容、真實評價及預測評價-----------------------")
display_test_Sentiment(3) # 顯示第三篇testin文章的內容、真實評價及預測評價

def predict_review(input_text):
    # 首先將"影評字串"轉成"數字list"
    input_seq = token.texts_to_sequences([input_text])

    # 接下來使用sequence.pad_sequences() 方法將數字list截長補短，使所有數字list的長度都為100
    pad_input_seq = sequence.pad_sequences(input_seq, maxlen=100)

    # 最後使用multilayer perceptron模型進行預測
    predict_result = model.predict_classes(pad_input_seq)

    print("預測結果：{0}".format(SentimentDict[predict_result[0][0]]))

print()
print("-----------------------輸入影評字串，並預測評價-----------------------")
s = input("影評")
predict_review(s)


