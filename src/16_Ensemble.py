import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation
from tensorflow.keras.layers import AveragePooling2D, GlobalAvgPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import History
from scipy.stats import mode
import math
import pickle
from sklearn.metrics import accuracy_score

# 資料預處理函數
def prepare_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()  # 載入 CIFAR-10 資料集

    # 計算訓練集的均值和標準差，用於資料標準化
    mean = np.mean(X_train, axis = (0, 1, 2, 3))
    std = np.std(X_train, axis = (0, 1, 2, 3))
    
    # 對訓練集和測試集進行標準化處理
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    # 將標籤轉換為 One-Hot 編碼形式
    y_test_label = np.ravel(y_test)  # 將二維數組展平為一維數組
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-Hot 編碼
    return X_train, X_test, y_train, y_test, y_test_label

# 定義卷積層函數
def make_convlayer(input, fsize, layers):
    x = input
    for i in range(layers):
        x = Conv2D(
            filters = fsize,            # 卷積核數量
            kernel_size = 3,            # 卷積核大小為 3x3
            padding = "same"            # 使用 'same' 填充以保持輸入和輸出的尺寸一致
        )(x)
        x = BatchNormalization()(x)     # 批量正則化，幫助穩定訓練
        x = Activation("relu")(x)       # ReLU 激活函數，提供非線性轉換
    return x

# 創建模型函數
def create_model():
    input = Input(shape = (32, 32, 3))          # 定義模型的輸入形狀
    x = make_convlayer(input, 64, 3)            # 添加卷積層區塊
    x = AveragePooling2D(2)(x)                  # 2x2 平均池化層
    x = make_convlayer(x, 128, 3)               # 添加另一個卷積層區塊
    x = AveragePooling2D(2)(x)                  # 2x2 平均池化層
    x = make_convlayer(x, 256, 3)               # 添加另一個卷積層區塊
    x = GlobalAvgPool2D()(x)                    # 全局平均池化層
    x = Dense(10, activation = "softmax")(x)    # 最終的全連接層，輸出 10 個類別
    model = Model(input, x)                     # 創建 Keras 模型
    return model

# 自定義的檢查點回調函數
class Checkpoint(Callback):
    def __init__(self, model, filepath):
        self._model = model                     # 使用另一個屬性名稱以避免衝突
        self.filepath = filepath
        self.best_val_acc = 0.0                 # 初始化最佳驗證準確度
    
    def on_epoch_end(self, epoch, logs):
        if logs['val_acc'] > self.best_val_acc:                         # 如果當前驗證準確度更好
            self._model.save_weights(self.filepath + ".weights.h5")     # 保存模型權重
            self.best_val_acc = logs['val_acc']                         # 更新最佳驗證準確度
            print('Weights saved.', self.best_val_acc)

# 進行模型集成的投票函數
def ensemble_majority(models, X):
    pred_labels = np.zeros((X.shape[0], len(models)))                       # 初始化預測標籤矩陣

    for i, model in enumerate(models):
        pred_labels[:, i] = np.argmax(model.predict(X), axis = 1)           # 獲取每個模型的預測結果(因為是Softmax)
    return np.ravel(mode(pred_labels, axis = 1)[0])                         # 根據投票結果獲得最終預測標籤

# 訓練函數
def train(X_train, X_test, y_train, y_test, y_test_label):
    models_num = 5                                      # 模型數量
    batch_size = 1024                                   # 批次大小
    epoch = 80                                          # 訓練的迭代次數
    models = []                                         # 存儲模型的列表
    history_all = {"hists": [], "ensemble_test": []}    # 儲存歷史和集成測試準確度(初始化空的哈希表)

    model_predict = np.zeros((X_test.shape[0], models_num))     # 初始化模型預測結果矩陣
    for i in range(models_num):                                 # Create 5個 Model
        print('Model', i + 1)
        train_model = create_model()                            # 創建模型
        train_model.compile(optimizer = 'adam',                 # 編譯模型
                            loss = 'categorical_crossentropy',
                            metrics = ["acc"])
        
        models.append(train_model)                          # 將模型添加到模型列表中
        hist = History()                                    # 初始化歷史回調
        cpont = Checkpoint(train_model, f'weights_{i}.h5')  # 初始化檢查點回調(自動保存weight)

        def step_decay(epoch):
            initial_lrate = 0.001                           # 初始學習率
            drop = 0.5                                      # 學習率衰減率
            epochs_drop = 10.0                              # 學習率衰減周期
            lrate = initial_lrate * math.pow(drop, 
                                             math.floor((1 + epoch) / epochs_drop))
            return lrate                                    # 計算當前學習率

        lrate = LearningRateScheduler(step_decay)                               # 初始化學習率調度器
        datagen = ImageDataGenerator(rotation_range = 15,
                                     width_shift_range = 0.1,
                                     height_shift_range = 0.1,
                                     horizontal_flip = True,
                                     zoom_range = 0.2)                          # 設置資料增強生成器
        # 訓練模型
        train_model.fit(datagen.flow(X_train, y_train, batch_size = batch_size),
                        epochs = epoch,
                        steps_per_epoch = X_train.shape[0] // batch_size,       # 每個 epoch 的步數
                        validation_data = (X_test, y_test),
                        verbose = 1,
                        callbacks = [hist, cpont, lrate])                       # 使用回調函數
        
        train_model.load_weights(f'weights_{i}.weights.h5')                     # 加載最佳權重

        for layer in train_model.layers:
            layer.trainable = False  # 凍結模型的權重
        
        model_predict[:, i] = np.argmax(train_model.predict(X_test), axis = -1)     # 獲取模型預測結果
        history_all['hists'].append(hist.history)                                   # 保存歷史紀錄
        ensemble_test_pred = ensemble_majority(models, X_test)                      # 獲取集成預測結果
        ensemble_test_acc = accuracy_score(y_test_label, ensemble_test_pred)        # 計算集成準確度
        history_all['ensemble_test'].append(ensemble_test_acc)                      # 保存集成準確度
        print('Current Ensemble Accuracy : ', ensemble_test_acc)
    
    history_all['corrcoef'] = np.corrcoef(model_predict, rowvar = False)            # 計算預測結果的相關係數
    print('Correlation predicted value')
    print(history_all['corrcoef'])                                                  # 輸出相關係數矩陣

# 實際執行
X_train, X_test, y_train, y_test, y_test_label = prepare_data()     # 準備數據
train(X_train, X_test, y_train, y_test, y_test_label)               # 執行訓練
