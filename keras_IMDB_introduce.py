import urllib.request
import os, re
import tarfile
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

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

print()
print("-----------------------印出第3個評論的字串，和對應的label-----------------------")
print(train_text[3])
print(train_label[3])

# 接下來要建立"字典"
token = Tokenizer(num_words=2000) # 建立一個有2000字的字典
token.fit_on_texts(train_text)
# 讀取所有的訓練資料影評，依照每一個英文字，在影評中出現的次數進行排序
# 排序的前2000名的英文字，會列入字典中

print()
print("-----------------------查看token讀取多少文章-----------------------")
print(token.document_count)

print()
print("-----------------------查看token.word_index-----------------------")
# word_index是dict資料形態，出現次數最多的會排最前面
print("the:{0} , is:{1} , high:{2}".format(token.word_index['the'], token.word_index['is'], token.word_index['high']))
# 'the' 這個字串對應到字典的value為1 代表最常出現
# 'is'  這個字串對應到字典的value為6 代表出現頻率為第六名
# 'high' 這個字串對應到字典的value為308  代表出現頻率為第308名

# 接下來將影評文字轉換成數字list
x_train_seq = token.texts_to_sequences(train_text) # 為一個有25000個元素的list，每個元素也是一個list，該list的元素則存放著轉換後的數字
x_test_seq = token.texts_to_sequences(test_text) # 為一個有25000個元素的list，每個元素也是一個list，該list的元素則存放著轉換後的數字
print()
print("-----------------------查看轉換前及轉換後的結果-----------------------")
print(train_text[0]) # 印出第0個影評文字 (string字串)
print(x_train_seq[0]) # 印出第0個影評文字轉換成數字後的結果 ( list 其中每個元素為數字)

# 接下來使用sequence.pad_sequences() 方法將數字list截長補短，使所有數字list的長度都為100
# 若數字list的長度=126，就將前面的26個數字截去
# 若數字list的長度為59，就在前面補上41個0
x_train = sequence.pad_sequences(x_train_seq, maxlen=100) # 為一個有25000個元素的ndarray，每個元素是一個list，該list的元素則存放著轉換後的數字 (現在強制變為100個)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100) # 為一個有25000個元素的ndarray，每個元素是一個list，該list的元素則存放著轉換後的數字 (現在強制變為100個)
print()
print("-----------------------查看截長補短的結果-----------------------")
print("before pad_sequences length={0}".format(len(x_train_seq[0])))
print(x_train_seq[0])
print("after pad_sequences length={0}".format(len(x_train[0])))
print(x_train[0])
print()
print("before pad_sequences length={0}".format(len(x_train_seq[6])))
print(x_train_seq[6])
print("after pad_sequences length={0}".format(len(x_train[6])))
print(x_train[6])
