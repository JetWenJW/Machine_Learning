import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History, LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 定義卷積層模型
def make_convlayer():
    model = Sequential()            # 創建 Sequential 模型

    # 添加第一個卷積層
    model.add(Conv2D(
        filters = 64,                 # 卷積核數量
        kernel_size = 3,              # 卷積核大小
        padding = 'same',             # 填充方式
        activation = 'relu',          # 激活函數
        input_shape = (32, 32, 3)     # 輸入形狀
    ))

    # 添加最大池化層
    model.add(MaxPooling2D(pool_size = 2))  # 池化大小

    # 添加第二個卷積層
    model.add(Conv2D(
        filters = 128,        # 卷積核數量
        kernel_size = 3,      # 卷積核大小
        padding = 'same',     # 填充方式
        activation = 'relu'   # 激活函數
    ))

    # 添加最大池化層
    model.add(MaxPooling2D(pool_size = 2))  # 池化大小
    
    # 添加第三個卷積層
    model.add(Conv2D(
        filters = 256,        # 卷積核數量
        kernel_size = 3,      # 卷積核大小
        padding = 'same',     # 填充方式
        activation = 'relu'   # 激活函數
    ))

    # 添加最大池化層
    model.add(MaxPooling2D(pool_size = 2))  # 池化大小
    
    # 添加展平層
    model.add(Flatten())

    # 添加 Dropout 層，防止過擬合
    model.add(Dropout(0.4))

    # 添加全連接層
    model.add(Dense(512, activation = 'relu'))  # 全連接層，512 個神經元，激活函數為 ReLU

    # 添加輸出層，使用 softmax 激活函數進行多類別分類
    model.add(Dense(10, activation = 'softmax'))  # 輸出層，10 個神經元，激活函數為 softmax

    # 編譯模型
    model.compile(loss = "categorical_crossentropy",
                  optimizer = optimizers.Adam(learning_rate = 0.001),
                  metrics = ["accuracy"])  # 損失函數為分類交叉熵，優化器為 Adam，評估指標為準確率
    
    return model  # 返回模型

# 定義學習率衰減函數
def step_decay(epoch):
    lrate = 0.001           # 初始學習率
    # 根據 epoch 進行學習率調整
    if epoch >= 50: lrate /= 5.0
    if epoch >= 100: lrate /= 5.0
    if epoch >= 150: lrate /= 5.0
    
    return lrate  # 返回調整後的學習率

# 訓練模型的函數
def train_batchsize(model, data, batch_size, epochs, decay):
    x_train, y_train, x_test, y_test = data  # 解包數據

    # 訓練數據增強生成器
    train_gen = ImageDataGenerator(
        rescale = 1.0 / 255.0,            # 將圖像像素值縮放到 [0, 1] 範圍
        width_shift_range = 0.1,          # 圖像寬度平移範圍
        height_shift_range = 0.1,         # 圖像高度平移範圍
        rotation_range = 10,              # 圖像旋轉範圍
        zoom_range = 0.1,                 # 圖像縮放範圍
        horizontal_flip = True            # 圖像水平翻轉
    ).flow(x_train, y_train, batch_size = batch_size)

    # 測試數據增強生成器
    test_gen = ImageDataGenerator(rescale = 1.0 / 255.0).flow(x_test, y_test, batch_size = 128)

    hist = History()  # 初始化 History 回調
    # 訓練模型
    model.fit(train_gen,
              steps_per_epoch = x_train.shape[0] // batch_size,   # 每個 epoch 的步數
              epochs = epochs,                                    # 訓練的總 epoch 數
              validation_data = test_gen,                         # 驗證數據
              callbacks = [hist, decay])                          # 使用回調函數
    
    return hist.history  # 返回訓練歷史記錄

# 訓練函數
def train(train_mode):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # 載入 CIFAR-10 資料集

    # 對標籤進行 One-Hot 編碼
    y_train = to_categorical(y_train)                       # 將訓練數據的標籤轉換為 One-Hot 編碼格式
    y_test = to_categorical(y_test)                         # 將測試數據的標籤轉換為 One-Hot 編碼格式

    data = (x_train, y_train, x_test, y_test)               # 準備數據元組，包含訓練和測試數據及其標籤
    
    model = make_convlayer()                                # 創建卷積層模型
    histories = []                                          # 用於保存歷史記錄的列表
    decay = LearningRateScheduler(step_decay)               # 創建學習率衰減調度器
    same_lr = LearningRateScheduler(lambda epoch: 0.001)    # 固定學習率調度器，學習率保持不變

    if train_mode == 0:
        # 訓練模型，使用學習率衰減
        histories.append(train_batchsize(
            model, data, batch_size = 128, epochs = 200, decay = decay  # 訓練模型並記錄歷史
        ))

    if train_mode == 1:
        # 訓練模型，使用不同的批次大小
        histories.append(train_batchsize(
            model, data, batch_size = 128, epochs = 50, decay = same_lr  # 使用固定學習率訓練模型
        ))
        histories.append(train_batchsize(
            model, data, batch_size = 640, epochs = 50, decay = same_lr  # 使用不同批次大小訓練模型
        ))
        histories.append(train_batchsize(
            model, data, batch_size = 3200, epochs = 50, decay = same_lr
        ))
        histories.append(train_batchsize(
            model, data, batch_size = 16000, epochs = 50, decay = same_lr
        ))
        
        # 合併歷史記錄
        joined_history = histories[0]                               # 初始化合併的歷史記錄
        for i in range(1, len(histories)):
            for key, value in histories[i].items():                 # 合併所有歷史記錄
                joined_history[key] = joined_history[key] + value
        return joined_history                                       # 返回合併的歷史記錄

# 執行訓練
history = train(0)          # 使用學習率衰減訓練模型
history_batch = train(1)    # 使用不同批次大小訓練模型

# 繪製訓練準確度圖
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(history['accuracy'], label='lr', linestyle='--')   # 繪製學習率衰減模式的訓練準確度
plt.plot(history_batch['accuracy'], label='batch')          # 繪製不同批次大小模式的訓練準確度
plt.legend()                                                # 顯示圖例
plt.grid()                                                  # 顯示網格線
plt.xlabel('Epoch')                                         # 設置 x 軸標籤
plt.ylabel('Accuracy')                                      # 設置 y 軸標籤

# 繪製驗證準確度圖
plt.figure(figsize=(15, 15))
plt.subplot(2, 1, 1)
plt.plot(history['val_accuracy'], label='lr', linestyle='--')   # 繪製學習率衰減模式的驗證準確度
plt.plot(history_batch['val_accuracy'], label='batch')          # 繪製不同批次大小模式的驗證準確度
plt.legend()                                                    # 顯示圖例
plt.grid()                                                      # 顯示網格線
plt.xlabel('Epoch')                                             # 設置 x 軸標籤
plt.ylabel('Validation Accuracy')                               # 設置 y 軸標籤

plt.show()                                                      # 顯示圖形
