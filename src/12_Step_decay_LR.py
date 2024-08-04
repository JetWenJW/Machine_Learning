import numpy as np                                  # 導入NumPy庫，用於數學計算
from tensorflow.keras.datasets import cifar10       # 從Keras導入CIFAR-10數據集
from tensorflow.keras.utils import to_categorical   # 從Keras導入to_categorical，用於將標籤轉換為one-hot編碼

def prepare_data():
    """
    讀取CIFAR-10數據集，進行標準化處理並將標籤轉換為one-hot編碼
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()              # 讀取CIFAR-10數據集
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')   # 將數據類型轉換為float32
    x_train, x_test = x_train / 255.0, x_test / 255.0                       # 將像素值標準化到0到1之間
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)       # 將標籤轉換為one-hot編碼

    return x_train, x_test, y_train, y_test  # 返回處理後的數據

from tensorflow.keras.models import Sequential                  # 從Keras導入Sequential模型
from tensorflow.keras.layers import Dense, Dropout, Flatten     # 從Keras導入Dense、Dropout和Flatten層
from tensorflow.keras.layers import Conv2D, MaxPooling2D        # 從Keras導入Conv2D和MaxPooling2D層
from tensorflow.keras import optimizers                         # 從Keras導入優化器模塊

def make_convlayer():
    """
    創建一個卷積神經網絡模型
    """
    model = Sequential()  # 創建Sequential模型

    # 添加第一個卷積層
    model.add(Conv2D(
        filters = 64,             # 卷積層的濾波器數量
        kernel_size = 3,          # 卷積核的大小
        padding = 'same',         # 使用same填充以保持輸出尺寸與輸入尺寸相同
        activation = 'relu',      # 卷積層的激活函數
        input_shape = (32, 32, 3) # 輸入數據的形狀（32x32像素，3個通道）
    ))

    model.add(MaxPooling2D(pool_size=2))  # 添加最大池化層，池化窗口為2x2

    # 添加第二個卷積層
    model.add(Conv2D(
        filters = 256,        # 卷積層的濾波器數量
        kernel_size = 3,      # 卷積核的大小
        padding = 'same',     # 使用same填充以保持輸出尺寸與輸入尺寸相同
        activation = 'relu'   # 卷積層的激活函數
    ))
    model.add(MaxPooling2D(pool_size = 2))              # 添加最大池化層，池化窗口為2x2
    model.add(Flatten())                                # 將多維輸出展平為一維數據
    model.add(Dropout(0.4))                             # 添加Dropout層，丟棄40%的神經元以防止過擬合
    model.add(Dense(512, activation='relu'))            # 全連接層，具有512個神經元和relu激活函數
    model.add(Dense(10, activation="softmax"))          # 最終輸出層，具有10個神經元和softmax激活函數（用於多類別分類）

    # 編譯模型
    model.compile(loss="categorical_crossentropy",                  # 設置損失函數為categorical_crossentropy
                  optimizer=optimizers.Adam(learning_rate=0.001),   # 設置優化器為Adam，學習率為0.001
                  metrics=["accuracy"])                             # 設置評估指標為準確率

    return model  # 返回創建的模型

import math  # 導入math庫，用於數學計算
from tensorflow.keras.preprocessing.image import ImageDataGenerator     # 從Keras導入ImageDataGenerator，用於數據增強
from tensorflow.keras.callbacks import LearningRateScheduler, Callback  # 從Keras導入LearningRateScheduler和Callback

class LRHistory(Callback):
    """
    回調類，用於記錄學習率和準確率
    """
    def on_train_begin(self, batch, logs={}):
        self.acc = []                               # 初始化準確率列表
        self.lr = []                                # 初始化學習率列表

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('accuracy'))       # 記錄每個epoch的準確率
        self.lr.append(step_decay(len(self.acc)))   # 記錄每個epoch的學習率

def step_decay(epoch):
    """
    學習率衰減函數
    """
    initial_lrate = 0.001               # 初始學習率
    drop = 0.5                          # 學習率衰減因子
    epochs_drop = 10.0                  # 每10個epoch進行一次學習率衰減
    lrate = initial_lrate * math.pow(drop, math.floor((epoch) / epochs_drop))  # 計算當前學習率
    return lrate                        # 返回學習率

def train(x_train, x_test, y_train, y_test):
    """
    訓練模型
    """
    model = make_convlayer()                    # 創建卷積神經網絡模型
    lr_history = LRHistory()                    # 創建學習率歷史回調對象
    lrate = LearningRateScheduler(step_decay)   # 創建學習率調度器回調對象
    call_backs_list = [lr_history, lrate]       # 回調列表

    # 創建ImageDataGenerator對象，進行數據增強
    datagen = ImageDataGenerator(width_shift_range = 0.1,     # 水平平移範圍
                                 height_shift_range = 0.1,    # 垂直平移範圍
                                 rotation_range = 10,         # 旋轉角度範圍
                                 zoom_range = 0.1,            # 隨機縮放範圍
                                 horizontal_flip = True)      # 隨機水平翻轉

    batch_size = 128  # 批次大小
    epochs = 100  # 訓練的總輪數

    # 訓練模型
    history = model.fit(datagen.flow(x_train, y_train,                      # 使用增強後的訓練數據進行模型訓練
                        batch_size = batch_size),
                        steps_per_epoch = x_train.shape[0] // batch_size,     # 每個epoch的步數
                        epochs = epochs,                                      # 訓練的總輪數
                        verbose = 1,                                          # 訓練過程的輸出詳情
                        validation_data = (x_test, y_test),                   # 設置驗證數據
                        callbacks = call_backs_list                           # 設置回調函數
                        )

    return history, lr_history  # 返回訓練歷史和學習率歷史

# 讀取並準備數據
x_train, x_test, y_train, y_test = prepare_data()

# 訓練模型
history, lr_history = train(x_train, x_test, y_train, y_test)

import matplotlib.pyplot as plt  # 導入Matplotlib庫，用於可視化

x_range = list(range(1, len(lr_history.lr) + 1))  # 計算x軸範圍（即epoch數）

# 創建圖形
plt.figure(figsize = (15, 10))        # 設置圖形大小
plt.subplots_adjust(wspace = 0.2)     # 設置子圖之間的空間

# 繪製準確率曲線
plt.subplot(2, 2, 1)  # 設置子圖位置
plt.plot(x_range, history.history['accuracy'], label='train', linestyle='--')   # 繪製訓練準確率
plt.plot(x_range, history.history['val_accuracy'], label='Val_Acc')             # 繪製驗證準確率
plt.legend()            # 顯示圖例
plt.grid()              # 顯示網格
plt.xlabel('Epoch')     # 設置x軸標籤
plt.ylabel('Acc')       # 設置y軸標籤

# 繪製學習率曲線
plt.subplot(2, 1, 2)  # 設置子圖位置
plt.plot(x_range, lr_history.lr, label='Learning Rate')  # 繪製學習率曲線
plt.legend()                    # 顯示圖例
plt.grid()                      # 顯示網格
plt.xlabel('Epoch')             # 設置x軸標籤
plt.ylabel('Learning Rate')     # 設置y軸標籤
plt.show()                      # 顯示圖形
