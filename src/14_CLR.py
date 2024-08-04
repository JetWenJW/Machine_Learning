import numpy as np                                          # 引入 numpy 用於數值運算
from tensorflow.keras.datasets import cifar10               # 引入 cifar10 資料集
from tensorflow.keras.utils import to_categorical           # 引入 to_categorical 用於進行 one-hot 編碼

def prepare_data():
    # 載入 CIFAR-10 資料集
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 將圖像數據轉換為浮點數，並進行正規化處理（將數值範圍從 0-255 轉換為 0-1）
    x_train, x_test = x_train.astype('float'), x_test.astype('float')
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # 將標籤轉換為 one-hot 編碼
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    
    return x_train, x_test, y_train, y_test

from tensorflow.keras.models import Sequential                  # 引入 Sequential 模型
from tensorflow.keras.layers import Dense, Dropout, Flatten     # 引入 Dense、Dropout、Flatten 層
from tensorflow.keras.layers import Conv2D, MaxPooling2D        # 引入 Conv2D、MaxPooling2D 層
from tensorflow.keras import optimizers                         # 引入優化器

def make_convlayer():
    model = Sequential()  # 初始化一個序列模型
    
    # 添加第一個卷積層，輸入形狀為 (32, 32, 3)，輸出 64 個特徵圖，卷積核大小為 3x3，激活函數為 ReLU
    model.add(Conv2D(
        filters = 64,               # 卷積核數量
        kernel_size = 3,            # 卷積核大小
        padding = 'same',           # 保持卷積後圖像大小不變
        activation = 'relu',        # 激活函數
        input_shape = (32, 32, 3)   # 輸入形狀
    ))

    # 添加 2x2 池化層
    model.add(MaxPooling2D(pool_size = 2))  # 池化層，池化區域大小為 2x2

    # 添加第二個卷積層，輸出 128 個特徵圖，卷積核大小為 3x3，激活函數為 ReLU
    model.add(Conv2D(
        filters = 128,              # 卷積核數量
        kernel_size = 3,            # 卷積核大小
        padding = 'same',           # 保持卷積後圖像大小不變
        activation = 'relu'         # 激活函數
    ))

    # 添加 2x2 池化層
    model.add(MaxPooling2D(pool_size = 2))  # 池化層，池化區域大小為 2x2

    # 添加第三個卷積層，輸出 256 個特徵圖，卷積核大小為 3x3，激活函數為 ReLU
    model.add(Conv2D(
        filters = 256,              # 卷積核數量
        kernel_size = 3,            # 卷積核大小
        padding = 'same',           # 保持卷積後圖像大小不變
        activation = 'relu'         # 激活函數
    ))

    # 添加 2x2 池化層
    model.add(MaxPooling2D(pool_size = 2))                      # 池化層，池化區域大小為 2x2

    # 將三維特徵圖展平為一維
    model.add(Flatten())                                        # 將多維特徵圖展平為一維向量
    
    # 添加 Dropout 層，防止過擬合，丟棄 40% 的神經元
    model.add(Dropout(0.4))                                     # Dropout 層，丟棄比例為 40%

    # 添加全連接層，輸出 512 個神經元，激活函數為 ReLU
    model.add(Dense(512, activation = 'relu'))                  # 全連接層，輸出 512 個神經元，激活函數為 ReLU

    # 添加輸出層，10 個神經元對應 CIFAR-10 的 10 個類別，激活函數為 softmax
    model.add(Dense(10, activation = 'softmax'))                # 輸出層，10 個神經元，激活函數為 softmax

    # 編譯模型，損失函數使用 categorical_crossentropy，優化器使用 Adam，評估指標為準確率
    model.compile(
        loss = "categorical_crossentropy",                      # 使用 categorical_crossentropy 作為損失函數
        optimizer = optimizers.Adam(learning_rate = 0.001),     # 使用 Adam 優化器，學習率為 0.001
        metrics = ["accuracy"]                                  # 評估指標為準確率
    )

    return model  # 返回構建好的模型


from tensorflow.keras.callbacks import Callback     # 引入 Callback 基類
from tensorflow.keras import backend                # 引入 Keras 後端

class CyclicLR(Callback):
    def __init__(self, lr_min, lr_max, step_size, mode, gamma = 0.99994):
        # 初始化學習率循環調度器
        self.lr_min = lr_min        # 學習率的最小值
        self.lr_max = lr_max        # 學習率的最大值
        self.step_size = step_size  # 一個週期內的步數
        self.mode = mode            # 學習率模式
        self.gamma = gamma          # 指數衰減率
        self.clr_iterations = 0.    # 學習率循環計數器
        self.trn_iterations = 0.    # 訓練次數計數器
        self.history = {}           # 記錄學習率和損失的歷史
        self._init_scale(gamma)     # 初始化縮放函數

    def _init_scale(self, gamma):
        # 根據模式選擇縮放函數
        if self.mode == 0:
            self.scale_fn = lambda x: 1.                    # 不變模式
            self.scale_mode = 'cycle'
        elif self.mode == 1:
            self.scale_fn = lambda x: 1 / (2. ** (x - 1))   # 半衰減模式
            self.scale_mode = 'cycle'
        elif self.mode == 2:
            self.scale_fn = lambda x: gamma ** (x)          # 指數衰減模式
            self.scale_mode = 'iterations'

    def clr(self):
        # 計算當前循環中的學習率
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))            # 計算當前循環次數
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)            # 計算當前階段位置

        if self.scale_mode == 'cycle':
            decay = np.maximum(0, (1 - x)) * self.scale_fn(cycle)                   # 計算循環內的學習率衰減
            return self.lr_min + (self.lr_max - self.lr_min) * decay                # 計算當前學習率
        else:
            decay = np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)     # 計算基於迭代次數的學習率衰減
            return self.lr_min + (self.lr_max - self.lr_min) * decay                # 計算當前學習率

    def on_train_begin(self, logs = {}):
        # 訓練開始時初始化學習率
        logs = logs or {}                                               # 初始化日志
        self.losses = []                                                # 初始化損失列表
        self.lr = []                                                    # 初始化學習率列表
        if self.clr_iterations == 0:
            backend.set_value(self.model.optimizer.lr, self.lr_min)     # 設置初始學習率
        else:
            backend.set_value(self.model.optimizer.lr, self.clr())      # 根據當前循環次數設置學習率

    def on_batch_end(self, batch, logs = None):
        # 每個批次結束時更新學習率
        logs = logs or {}           # 初始化日志
        self.trn_iterations += 1    # 增加訓練次數
        self.clr_iterations += 1    # 增加學習率循環次數

        # 記錄學習率和訓練次數
        self.history.setdefault('learning_rate', []).append(backend.get_value(self.model.optimizer.lr))     # 記錄學習率
        self.history.setdefault('iterations', []).append(self.trn_iterations)                               # 記錄訓練次數
        for k,v in logs.items():
            self.history.setdefault(k, []).append(v)            # 記錄其他日志信息
        backend.set_value(self.model.optimizer.lr, self.clr())  # 更新學習率


from tensorflow.keras.preprocessing.image import ImageDataGenerator     # 引入 ImageDataGenerator 用於數據增強
from tensorflow.keras.callbacks import LearningRateScheduler, Callback  # 引入 LearningRateScheduler 和 Callback

def train(x_train, x_test, y_train, y_test, mode = 0):
    batch_size = 128                        # 設置批次大小為 128
    iteration = 50000                       # 總迭代次數為 50000
    stepsize = iteration / 128 * 4          # 計算步長，這裡為 50000 / 128 * 4
    lr_min = 0.0001                         # 設置最小學習率為 0.0001
    lr_max = 0.001                          # 設置最大學習率為 0.001

    # 創建 CyclicLR 回調函數，傳入循環模式、最小學習率、最大學習率和步長
    clr_triangular = CyclicLR(mode = mode,
                              lr_min = lr_min,
                              lr_max = lr_max,
                              step_size = stepsize)
    callbacks_list = [clr_triangular]       # 將 CyclicLR 回調函數加入回調列表
    
    # 創建模型
    model = make_convlayer()
    
    # 創建數據增強生成器
    datagen = ImageDataGenerator(width_shift_range = 0.1,     # 寬度平移範圍
                                 height_shift_range = 0.1,    # 高度平移範圍
                                 rotation_range = 10,         # 旋轉範圍
                                 zoom_range = 0.1,            # 縮放範圍
                                 horizontal_flip = True)      # 水平翻轉
    
    epochs = 100  # 訓練周期設置為 100
    
    # 訓練模型
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size = batch_size),  # 使用數據增強生成器生成的批次數據進行訓練
        steps_per_epoch = x_train.shape[0] // batch_size,         # 每個周期的步數
        epochs = epochs,                                          # 訓練周期數
        verbose = 1,                                              # 訓練過程顯示進度條
        validation_data = (x_test, y_test),                       # 驗證數據集
        callbacks = callbacks_list                                # 回調函數列表
    )
    
    return history, clr_triangular  # 返回訓練歷史和 CyclicLR 回調函數的實例

# 準備數據
x_train, x_test, y_train, y_test = prepare_data()

# 訓練模型，使用不同的循環學習率模式
history_0, clr_triangular_0 = train(x_train, x_test, y_train, y_test, mode = 0)  # 使用模式 0 訓練
history_1, clr_triangular_1 = train(x_train, x_test, y_train, y_test, mode = 1)  # 使用模式 1 訓練
history_2, clr_triangular_2 = train(x_train, x_test, y_train, y_test, mode = 2)  # 使用模式 2 訓練

import matplotlib.pyplot as plt  # 引入 matplotlib.pyplot 用於繪製圖像

# 繪製不同循環學習率模式下的準確率
plt.figure(figsize = (10, 5))                                                           # 設置圖形大小為 10x5
plt.plot(history_0.history['accuracy'], label = 'CLR', linestyle = '--')                # 繪製 CLR 模式下的準確率曲線，虛線表示
plt.plot(history_1.history['accuracy'], label = 'CLR, Half Decay', linestyle = '-.')    # 繪製 CLR 半衰減模式下的準確率曲線，點劃線表示
plt.plot(history_2.history['accuracy'], label = 'CLR, Exp Decay')                       # 繪製 CLR 指數衰減模式下的準確率曲線，實線表示
plt.legend()            # 顯示圖例
plt.grid()              # 添加網格線
plt.xlabel('Epoch')     # 設置 x 軸標籤為 Epoch
plt.ylabel('Train_Acc') # 設置 y 軸標籤為 Train_Acc
plt.show()              # 顯示圖形

# 繪製學習率變化情況
plt.figure(figsize = (10, 15))                                                     # 設置圖形大小為 10x15
plt.subplot(3, 1, 1)                                                               # 創建一個 3 行 1 列的子圖，並激活第一個子圖
plt.plot(clr_triangular_0.history['learning_rate'], label = 'Learning Rate(CLR)')  # 繪製 CLR 模式下的學習率變化曲線
plt.legend()                    # 顯示圖例
plt.grid()                      # 添加網格線
plt.xlabel('Iterations')        # 設置 x 軸標籤為 Iterations
plt.ylabel('Learning Rate')     # 設置 y 軸標籤為 Learning Rate
plt.show()                      # 顯示圖形
