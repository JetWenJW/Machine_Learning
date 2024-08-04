import numpy as np                                      # 導入NumPy庫，用於數學計算
import pandas as pd                                     # 導入Pandas庫，用於數據處理
from sklearn.model_selection import KFold               # 從sklearn導入KFold，用於交叉驗證
from tensorflow.keras.utils import to_categorical       # 從Keras導入to_categorical，用於將標籤轉換為one-hot編碼

# 讀取訓練數據集
train = pd.read_csv('./digit-recognizer/train.csv')     # 讀取CSV文件中的訓練數據
train_x = train.drop(['label'], axis=1)                 # 提取特徵數據，去掉標籤列
train_y = train['label']                                # 提取標籤數據
test_x = pd.read_csv('./digit-recognizer/test.csv')     # 讀取測試數據集（目前未使用）

# 使用KFold進行數據分割
kf = KFold(n_splits=4, shuffle=True, random_state=123)          # 設置KFold交叉驗證，將數據分為4份，並進行隨機打亂
tr_idx, va_idx = list(kf.split(train_x))[0]                     # 獲取第一折的訓練和驗證索引
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]         # 根據索引分割訓練數據和驗證數據
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]         # 根據索引分割訓練標籤和驗證標籤

# 標準化數據並重塑數據形狀以符合CNN的輸入格式
tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)  # 將像素值標準化到0到1之間

tr_x = tr_x.reshape(-1, 28, 28, 1)  # 將訓練數據重塑為(樣本數, 高度, 寬度, 通道數)格式
va_x = va_x.reshape(-1, 28, 28, 1)  # 將驗證數據重塑為相同格式

# 將標籤轉換為one-hot編碼
tr_y = to_categorical(tr_y, 10)  # 將訓練標籤轉換為10類的one-hot編碼
va_y = to_categorical(va_y, 10)  # 將驗證標籤轉換為10類的one-hot編碼

from tensorflow.keras.models import Sequential                  # 從Keras導入Sequential模型
from tensorflow.keras.layers import Dense, Dropout, Flatten     # 從Keras導入Dense、Dropout和Flatten層
from tensorflow.keras.layers import Conv2D, MaxPooling2D        # 從Keras導入Conv2D和MaxPooling2D層

# 創建一個Sequential模型
model = Sequential()

# 添加第一個卷積層
model.add(Conv2D(
        filters=32,                     # 卷積層的濾波器數量
        kernel_size=(5, 5),             # 卷積核的大小
        padding='same',                 # 使用same填充以保持輸出尺寸與輸入尺寸相同
        activation='relu',              # 卷積層的激活函數
        input_shape=(28, 28, 1)         # 輸入數據的形狀
))

# 添加第二個卷積層
model.add(Conv2D(
        filters=64,             # 卷積層的濾波器數量
        kernel_size=(7, 7),     # 卷積核的大小
        padding='same',         # 使用same填充以保持輸出尺寸與輸入尺寸相同
        activation='relu'       # 卷積層的激活函數
))

# 添加最大池化層
model.add(MaxPooling2D(pool_size=(2, 2)))       # 最大池化層，池化窗口為2x2
model.add(Dropout(0.5))                         # 添加Dropout層，丟棄50%的神經元以防止過擬合

# 添加第三個卷積層
model.add(Conv2D(
            filters=64,                 # 卷積層的濾波器數量
            kernel_size=(5, 5),         # 卷積核的大小
            padding='same',             # 使用same填充以保持輸出尺寸與輸入尺寸相同
            activation='relu'           # 卷積層的激活函數
))

# 添加第四個卷積層
model.add(Conv2D(
            filters=32,                 # 卷積層的濾波器數量
            kernel_size=(3, 3),         # 卷積核的大小
            padding='same',             # 使用same填充以保持輸出尺寸與輸入尺寸相同
            activation='relu'           # 卷積層的激活函數
))
model.add(MaxPooling2D(pool_size=(2, 2)))       # 添加最大池化層，池化窗口為2x2
model.add(Dropout(0.55))                        # 添加Dropout層，丟棄55%的神經元以防止過擬合

model.add(Flatten())                            # 將多維輸出展平為一維數據

# 添加全連接層
model.add(Dense(700, activation='relu'))        # 全連接層，具有700個神經元和relu激活函數
model.add(Dropout(0.3))                         # 添加Dropout層，丟棄30%的神經元以防止過擬合

model.add(Dense(150, activation='relu'))        # 全連接層，具有150個神經元和relu激活函數
model.add(Dropout(0.35))                        # 添加Dropout層，丟棄35%的神經元以防止過擬合

model.add(Dense(10, activation="softmax"))      # 最終輸出層，具有10個神經元和softmax激活函數（用於多類別分類）

# 編譯模型
model.compile(loss='categorical_crossentropy',  # 設置損失函數為categorical_crossentropy
              optimizer='adam',                 # 設置優化器為Adam
              metrics=["accuracy"])             # 設置評估指標為準確率

batch_size = 100        # 設置批次大小
epochs = 20             # 設置訓練輪數

# 訓練模型
history = model.fit(tr_x, tr_y,                         # 使用訓練數據進行模型訓練
                    batch_size=batch_size,              # 批次大小
                    epochs=epochs,                      # 訓練的總輪數
                    verbose=1,                          # 訓練過程的輸出詳情
                    validation_data=(va_x, va_y))       # 設置驗證數據

import matplotlib.pyplot as plt  # 導入Matplotlib庫，用於可視化

# 可視化訓練和驗證損失以及準確率
plt.figure(figsize=(15, 6))             # 設置圖形大小
plt.subplots_adjust(wspace=0.2)         # 設置子圖之間的空間

# 繪製損失曲線
plt.subplot(1, 2, 1)                    # 設置子圖位置
plt.plot(history.history['loss'],       # 繪製訓練損失
         label='training',              # 標籤為'training'
         linestyle='--')                # 設置線條樣式

plt.plot(history.history['val_loss'],   # 繪製驗證損失
         label='validation')            # 標籤為'validation'

plt.ylim(0, 1)          # 設置y軸範圍
plt.legend()            # 顯示圖例
plt.grid()              # 顯示網格
plt.xlabel('Epoch')     # 設置x軸標籤
plt.ylabel('loss')      # 設置y軸標籤

# 繪製準確率曲線
plt.subplot(1, 2, 2)                    # 設置子圖位置
plt.plot(history.history['accuracy'],   # 繪製訓練準確率
         linestyle='--',                # 設置線條樣式
         label='training')              # 標籤為'training'

plt.plot(history.history['val_accuracy'],       # 繪製驗證準確率
         label='validation')                    # 標籤為'validation'

plt.ylim(0.5, 1)                # 設置y軸範圍
plt.legend()                    # 顯示圖例
plt.grid()                      # 顯示網格
plt.xlabel('Epoch')             # 設置x軸標籤
plt.ylabel('acc')               # 設置y軸標籤

plt.show()  # 顯示圖形
