# 載入所需的套件
import numpy as np
import pandas as pd
import os

# 列出digit-recognizer目錄下的所有檔案
for dirname, _, filenames in os.walk('./digit-recognizer'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 讀取訓練資料集
train = pd.read_csv('./digit-recognizer/train.csv')
# 顯示訓練資料集
print(train)

# 載入Keras的to_categorical函數，將標籤轉為one-hot編碼
from tensorflow.keras.utils import to_categorical
# 載入KFold交叉驗證工具
from sklearn.model_selection import KFold

# 重新讀取訓練資料集
train = pd.read_csv('./digit-recognizer/train.csv')
# 分離特徵和標籤
train_x = train.drop(['label'], axis = 1)
train_y = train['label']
# 讀取測試資料集
test_x = pd.read_csv('./digit-recognizer/test.csv')

# 使用KFold進行交叉驗證，分為4份，打亂數據，隨機種子為123
kf = KFold(n_splits=4, shuffle=True, random_state=123)
# 獲取第一組訓練和驗證索引(含有編號0 ~ 3的驗證fold索引)
tr_idx, va_idx = list(kf.split(train_x))[0]
# 根據索引分割訓練和驗證數據
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 將特徵數據標準化到0-1之間
tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)

# 將標籤數據轉為one-hot編碼
tr_y = to_categorical(tr_y, 10)
va_y = to_categorical(va_y, 10)

# 打印訓練和驗證數據的形狀
print(tr_x.shape)
print(tr_y.shape)
print(va_x.shape)
print(va_y.shape)

# 計算訓練資料中每個標籤的數量
from collections import Counter

count = Counter(train['label'])
print(f'Train_Count: {count}')

# 繪製標籤分佈圖
import seaborn as sns

sns.countplot(train['label'])
# 打印第一張訓練圖片的數據
print(tr_x[0])

# 繪製訓練資料集中前50張圖片
import matplotlib.pyplot as plt

plt.figure(figsize = (12, 10))
x, y = 10, 5
for i in range(50):
    plt.subplot(y, x, i + 1)
    plt.imshow(tr_x[i].reshape((28, 28)), interpolation = 'nearest')
plt.show()

# 構建神經網路模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# 建立一個序列模型
model = Sequential()
# 添加一個全連接層，包含128個神經元，使用sigmoid激活函數，輸入維度為訓練資料的特徵數
model.add(Dense(128, input_dim = tr_x.shape[1], activation = 'sigmoid'))
# 添加輸出層，包含10個神經元，使用softmax激活函數
model.add(Dense(10, activation = 'softmax'))
# 編譯模型，損失函數為categorical_crossentropy，優化器為adam，評估指標為準確度
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# 顯示模型摘要
model.summary()

# 訓練模型，使用訓練資料進行5個訓練週期，批次大小為100，驗證資料為驗證集
hist = model.fit(tr_x, tr_y, 
                   epochs = 5, 
                   batch_size = 100, 
                   validation_data = (va_x, va_y), 
                   verbose = 1)

# 使用模型對測試資料進行預測
result = model.predict(test_x)
# 打印前5個預測結果
print(result[:5])
# 打印前5個預測結果的標籤
print([x.argmax() for x in result[:5]])
# 將預測結果轉為標籤
y_test = [x.argmax() for x in result]

# 讀取樣本提交檔案
submit_df = pd.read_csv('./digit-recognizer/sample_submission.csv')

# 顯示提交檔案的前5筆資料
submit_df.head()
# 將預測標籤填入提交檔案
submit_df['Label'] = y_test
# 顯示提交檔案的前5筆資料
submit_df.head()

# 將提交檔案保存為CSV格式
submit_df.to_csv('submission.csv', index = False)
