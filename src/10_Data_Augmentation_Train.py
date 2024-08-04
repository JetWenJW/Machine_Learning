import numpy as np                                  # 導入NumPy庫，用於數學運算
import pandas as pd                                 # 導入Pandas庫，用於數據處理
from sklearn.model_selection import KFold           # 從sklearn導入KFold，用於交叉驗證
from tensorflow.keras.utils import to_categorical   # 從Keras導入to_categorical，用於將標籤轉換為one-hot編碼

# 載入訓練數據
train = pd.read_csv('./digit-recognizer/train.csv')     # 讀取訓練數據集
train_x = train.drop(['label'], axis=1)                 # 提取特徵數據
train_y = train['label']                                # 提取標籤數據

# 載入測試數據（此處測試數據讀取錯誤，應改為測試集檔案）
test_x = pd.read_csv('./digit-recognizer/test.csv')  # 讀取測試數據集

# 使用KFold進行數據分割，將數據分成4份，並打亂數據
kf = KFold(n_splits=4, shuffle=True, random_state=123)
tr_idx, va_idx = list(kf.split(train_x))[0]                 # 獲取第一份分割的訓練和驗證索引
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]     # 分割訓練和驗證數據
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]     # 分割訓練和驗證標籤

# 標準化像素值至0到1之間
tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)  # 將像素值除以255進行標準化

# 重塑數據以符合CNN的輸入格式
tr_x = tr_x.reshape(-1, 28, 28, 1)  # 將訓練數據重塑為(樣本數, 高度, 寬度, 通道數)格式
va_x = va_x.reshape(-1, 28, 28, 1)  # 將驗證數據重塑為相同格式

# 將標籤轉換為one-hot編碼格式
tr_y = to_categorical(tr_y, 10)  # 將訓練標籤轉換為one-hot編碼
va_y = to_categorical(va_y, 10)  # 將驗證標籤轉換為one-hot編碼

from tensorflow.keras.models import Sequential                  # 從Keras導入Sequential模型
from tensorflow.keras.layers import Dense, Dropout, Flatten     # 從Keras導入Dense、Dropout和Flatten層
from tensorflow.keras.layers import Conv2D, MaxPooling2D        # 從Keras導入Conv2D和MaxPooling2D層

# 創建一個Sequential模型
model = Sequential()

# 添加卷積層
model.add(Conv2D(
    filters=32,                 # 卷積層的濾波器數量
    kernel_size=(5, 5),         # 卷積核的大小
    padding='same',             # 使用same填充以保持輸出尺寸
    activation='relu',          # 使用ReLU激活函數
    input_shape=(28, 28, 1)     # 輸入數據的形狀
))

model.add(Conv2D(
    filters=64,             # 卷積層的濾波器數量
    kernel_size=(7, 7),     # 卷積核的大小
    padding='same',         # 使用same填充以保持輸出尺寸
    activation='relu'       # 使用ReLU激活函數
))

model.add(Dropout(0.5))  # 添加Dropout層以防止過擬合，丟棄50%的神經元

model.add(Conv2D(
    filters=64,             # 卷積層的濾波器數量
    kernel_size=(5, 5),     # 卷積核的大小
    padding='same',         # 使用same填充以保持輸出尺寸
    activation='relu'       # 使用ReLU激活函數
))

model.add(Conv2D(
    filters=32,             # 卷積層的濾波器數量
    kernel_size=(3, 3),     # 卷積核的大小
    padding='same',         # 使用same填充以保持輸出尺寸
    activation='relu'       # 使用ReLU激活函數
))

model.add(MaxPooling2D(pool_size=(2, 2)))  # 添加最大池化層，池化窗口為2x2

model.add(Dropout(0.55))    # 添加Dropout層以防止過擬合，丟棄55%的神經元

model.add(Flatten())        # 將多維輸出展平成一維數據

# 添加全連接層
model.add(Dense(700, activation='relu'))    # 全連接層，具有700個神經元和ReLU激活函數
model.add(Dropout(0.3))                     # 添加Dropout層以防止過擬合，丟棄30%的神經元

model.add(Dense(150, activation='relu'))    # 全連接層，具有150個神經元和ReLU激活函數
model.add(Dropout(0.35))                    # 添加Dropout層以防止過擬合，丟棄35%的神經元

model.add(Dense(10, activation='softmax'))  # 最終輸出層，具有10個神經元和softmax激活函數（用於分類）

# 編譯模型
momentum = 0.5  # 設置動量（目前在此代碼中未使用）
model.compile(loss='categorical_crossentropy',  # 設置損失函數為categorical_crossentropy
              optimizer='adam',                 # 設置優化器為Adam
              metrics=['accuracy'])             # 設置評估指標為準確率

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 從Keras導入ImageDataGenerator，用於數據增強

# 創建ImageDataGenerator實例，進行數據增強
datagen = ImageDataGenerator(width_shift_range=0.1,     # 隨機水平位移範圍
                             height_shift_range=0.1,    # 隨機垂直位移範圍
                             rotation_range=10,         # 隨機旋轉範圍
                             zoom_range=0.1)            # 隨機縮放範圍

batch_size = 100  # 設置批次大小為100
epochs = 20  # 設置訓練輪數為20

# 訓練模型
history = model.fit(datagen.flow(tr_x, tr_y, batch_size=batch_size),    # 使用數據增強進行訓練
                    steps_per_epoch=tr_x.shape[0] // batch_size,        # 每個訓練輪的步數
                    epochs=epochs,                                      # 訓練的總輪數
                    verbose=1,                                          # 設置訓練過程的輸出詳情
                    validation_data=(va_x, va_y))                       # 設置驗證數據
