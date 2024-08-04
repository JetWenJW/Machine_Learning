import numpy as np                                  # 引入數據處理模組numpy
from tensorflow.keras.datasets import cifar10       # 從keras中引入CIFAR-10數據集
from tensorflow.keras.utils import to_categorical   # 引入將類別編碼轉換為one-hot編碼的工具

def prepare_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()  # 載入CIFAR-10數據集，分為訓練集和測試集
    mean = np.mean(X_train, axis = (0, 1, 2, 3))                # 計算訓練集的均值
    std = np.std(X_train, axis = (0, 1, 2, 3))                  # 計算訓練集的標準差

    x_train = (X_train - mean) / (std + 1e-7)   # 標準化訓練集
    x_test = (X_test - mean) / (std + 1e-7)     # 標準化測試集

    y_test_label = np.ravel(y_test)     # 將測試標籤展平成一維
    y_train = to_categorical(y_train)   # 將訓練標籤轉換為one-hot編碼
    y_test = to_categorical(y_test)     # 將測試標籤轉換為one-hot編碼

    return X_train, X_test, y_train, y_test, y_test_label  # 返回處理好的數據集和標籤

from tensorflow.keras.layers import Input, Conv2D, Dense, Activation    # 引入必要的層
from tensorflow.keras.layers import AveragePooling2D, GlobalAvgPool2D   # 引入池化層
from tensorflow.keras.layers import BatchNormalization                  # 引入批量正則化層
from tensorflow.keras import regularizers                               # 引入正則化器
from tensorflow.keras.models import Model                               # 引入模型類

"""
basic_conv_block1()
basic_conv_block2()
建立卷積層

Parameters: 
    inp(Input): 輸入層
    fsize(int): 過濾器大小
    layers(int) : 層的數量
Returns: Conv2D物件
"""

def basic_conv_block1(inp, fsize, layers):
    x = inp                             # 輸入張量，通常來自模型的前一層
    for i in range(layers):             # 在這個區塊中進行多次迭代
        x = Conv2D(
            filters = fsize,            # 卷積核的數量（即輸出通道數）
            kernel_size = 3,            # 卷積核的尺寸為 3x3
            padding = "same"            # 使用 'same' 填充，以保持輸入和輸出的尺寸一致
        )(x)                            # 應用卷積層
        x = BatchNormalization()(x)     # 應用批量正則化層，這有助於穩定訓練過程
        x = Activation("relu")(x)       # 應用 ReLU 激活函數，為非線性轉換提供支持
    return x                            # 返回經過處理的張量

def basic_conv_block2(inp, fsize, layers):
    weight_decay = 1e-4                                         # L2 正則化的強度參數
    x = inp                                                     # 輸入張量
    for i in range(layers):                                     # 在這個區塊中迭代多次
        x = Conv2D(
            filters = fsize,                                    # 卷積核數量
            kernel_size = 3,                                    # 卷積核大小
            padding = "same",                                   # 使用 'same' 填充，保持輸入和輸出的尺寸相同
            kernel_regularizer = regularizers.l2(weight_decay)  # 應用 L2 正則化
        )(x)                                                    # 應用卷積層
        x = BatchNormalization()(x)                             # 應用批量正則化
        x = Activation('relu')(x)                               # 應用 ReLU 激活函數
    return x                                                    # 返回經過處理的張量

def create_cnn(model_num):
    """
    建立模型
    Parameters: 
        model_num(int): 模型編號
    Returns: 模型
    """    
    inp = Input(shape = (32, 32, 3))                    # 定義輸入層的形狀
    if model_num < 5:
        x = basic_conv_block1(inp, 64, 3)               # 添加卷積區塊1
        x = AveragePooling2D(2)(x)                      # 添加平均池化層

        x = basic_conv_block1(x, 128, 3)                # 添加第二個卷積區塊
        x = AveragePooling2D(2)(x)                      # 添加平均池化層

        x = basic_conv_block1(x, 256, 3)                # 添加第三個卷積區塊
        x = GlobalAvgPool2D()(x)                        # 添加全局平均池化層
        x = Dense(10, activation = 'softmax')(x)        # 添加全連接層並使用softmax激活
        model = Model(inp, x)                           # 建立模型
    else:
        x = basic_conv_block2(inp, 64, 3)               # 添加卷積區塊1
        x = AveragePooling2D(2)(x)                      # 添加平均池化層

        x = basic_conv_block2(x, 128, 3)                # 添加第二個卷積區塊
        x = AveragePooling2D(2)(x)                      # 添加平均池化層

        x = basic_conv_block2(x, 256, 3)                # 添加第三個卷積區塊
        x = GlobalAvgPool2D()(x)                        # 添加全局平均池化層
        x = Dense(10, activation = 'softmax')(x)        # 添加全連接層並使用softmax激活
        model = Model(inp, x)                           # 建立模型
    return model                                        # 返回建立好的模型

def ensemble_average(models, X):
    """集合平均
    Parameters: 
        models(list): 模型列表
        X(array): 驗證資料
    Returns : 個圖像的預測值
    """
    preds_sum = None                        # 初始化預測總和
    for model in models:                    # 遍歷所有模型
        if preds_sum is None:               # 如果預測總和為空
            preds_sum = model.predict(X)    # 將第一個模型的預測值賦給預測總和
        else:
            preds_sum += model.predict(X)   # 將後續模型的預測值加到預測總和中
    probs = preds_sum / len(models)         # 計算平均預測值
    return np.argmax(probs, axis = 1)       # 返回每個圖像的預測類別

from tensorflow.keras.callbacks import Callback  # 引入回調類
class Checkpoint(Callback):
    
    def __init__(self, model, filepath):
        """
        Parameters:
            model(Model): 訓練中的模型
            filepath(str): 儲存權重的資料夾路徑
            best_val_acc(int): 目前最高準確率
        """
        self._model = model                 # 使用另一個屬性名稱避免衝突
        self.filepath = filepath
        self.best_val_acc = 0.0             # 初始化最高準確率

    def on_epoch_end(self, epoch, logs):
        """
        重新定義訓練週期結束時所呼出的函式
        從剛剛的訓練週期當中儲存準確率較高的權重
        Parameters:
            epoch(int): 訓練次數
            logs(dict): {'val_acc': 損失 , 'val_acc': 準確率 }
        """
        if self.best_val_acc < logs['val_acc']:                         # 如果目前的準確率高於最高準確率
            self._model.save_weights(self.filepath + ".weights.h5")     # 儲存權重            
            self.best_val_acc = logs['val_acc']                         # 更新最高準確率
            print('Weight saved.', self.best_val_acc)                   # 輸出保存提示信息

import math                                                             # 引入數學模組
import pickle                                                           # 引入pickle模組，用於序列化
from sklearn.metrics import accuracy_score                              # 引入計算準確率的工具
from tensorflow.keras.preprocessing.image import ImageDataGenerator     # 引入圖像數據生成器
from tensorflow.keras.callbacks import LearningRateScheduler            # 引入學習率調度器
from tensorflow.keras.callbacks import History                          # 引入訓練歷史記錄回調

def train(X_train, X_test, y_train, y_test, y_test_label):
    n_estimators = 9                    # 定義模型數量
    batch_size = 1024                   # 定義批量大小
    epoch = 80                          # 定義訓練週期數
    models = []                         # 初始化模型列表
    global_hist = {"hists":[], "ensemble_test":[]}              # 初始化全局歷史記錄字典
    single_preds = np.zeros((X_test.shape[0], n_estimators))    # 初始化單個模型預測值矩陣
    for i in range(n_estimators):
        print('Model', i + 1)                   # 輸出模型編號
        train_model = create_cnn(i)             # 創建模型
        train_model.compile(optimizer = 'adam',
                            loss = 'categorical_crossentropy',
                            metrics = ["acc"])  # 編譯模型
        models.append(train_model)              # 將模型添加到列表中

        hist = History()                                # 初始化歷史記錄回調
        cp = Checkpoint(train_model, f'weight_{i}.h5')  # 初始化檢查點回調

        def step_decay(epoch):
            initial_lrate = 0.001                   # 初始學習率
            drop = 0.5                              # 學習率下降因子
            epochs_drop = 10.0                      # 每隔多少個週期下降一次
            lrate = initial_lrate * math.pow(drop, 
                                             math.floor((1 + epoch) / epochs_drop))  # 計算新的學習率
            return lrate
        
        lrate = LearningRateScheduler(step_decay)               # 初始化學習率調度器
        datagen = ImageDataGenerator(rotation_range = 15,       # 初始化數據生成器
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     horizontal_flip = True,
                                     zoom_range = 0.2)
        train_model.fit(datagen.flow(X_train, y_train, batch_size = batch_size),
                        epochs = epoch,
                        steps_per_epoch = X_train.shape[0] // batch_size,
                        validation_data = (X_test, y_test),
                        verbose = 1,
                        callbacks = [hist, cp, lrate])          # 訓練模型
        
        train_model.load_weight(f'weights_{i}.weights.h5')      # 加載最佳權重

        for layers in train_model.layers:  # 冻结所有層
            layers.trainable = False
        
        single_preds[:, i] = np.argmax(train_model.predict(X_test), axis = -1)  # 儲存單個模型的預測結果
        global_hist['hists'].append(hist.history)                               # 儲存訓練歷史記錄

        ensemble_test_pred = ensemble_average(models, X_test)                   # 計算集合模型的預測結果
        ensemble_test_acc = accuracy_score(y_test_label, ensemble_test_pred)    # 計算集合模型的準確率

        global_hist['ensemble_test'].append(ensemble_test_acc)                  # 儲存集合模型的準確率
        print('Current Ensemble Test Accuracy : ', ensemble_test_acc)           # 輸出目前集合模型的準確率

    global_hist['corrcoef'] = np.corrcoef(single_preds, rowvar = False)         # 計算單個模型預測結果的相關係數
    print('Correlation predicted value')                                        # 輸出相關係數提示信息
    print(global_hist['corrcoef'])                                              # 輸出相關係數矩陣

X_train, X_test, y_train, y_test, y_test_label = prepare_data()                 # 準備數據
train(X_train, X_test, y_train, y_test, y_test_label)                           # 訓練模型
