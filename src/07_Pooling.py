from hyperopt import hp, Trials, tpe, fmin, STATUS_OK
# 從hyperopt導入需要的模組：
# - hp：用於定義超參數空間
# - Trials：用於記錄實驗結果
# - tpe：樹結構的Parzen估計（TPE）演算法，用於超參數優化
# - fmin：進行超參數優化的函數
# - STATUS_OK：用於指示優化過程中的狀態
from tensorflow.keras.utils import to_categorical   # 從tensorflow.keras.utils導入to_categorical，用於將標籤轉換為one-hot編碼格式
from tensorflow.keras.models import Sequential      # 從tensorflow.keras.models導入Sequential模型，用於構建模型的序列式堆疊
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# 從tensorflow.keras.layers導入以下層：
# - Conv2D：卷積層
# - MaxPooling2D：最大池化層
# - Dropout：Dropout層，用於防止過擬合
# - Flatten：展平層，將多維輸出展平成一維
# - Dense：全連接層
import numpy as np                              # 導入NumPy庫，用於數學運算和數據處理
import pandas as pd                             # 導入Pandas庫，用於數據處理和操作
from sklearn.model_selection import KFold       # 從sklearn.model_selection導入KFold，用於交叉驗證，將數據分為多個折


def prepare_data():
    # 載入數據
    train = pd.read_csv('./digit-recognizer/train.csv')     # 讀取訓練數據集
    train_x = train.drop(['label'], axis=1)                 # 提取特徵數據
    train_y = train['label']                                # 提取標籤數據
    test_x = pd.read_csv('./digit-recognizer/test.csv')     # 讀取測試數據集（代碼中此處應修正為測試數據集檔案）

    # 使用KFold進行數據分割
    kf = KFold(n_splits=4, shuffle=True, random_state=123)      # 設置KFold交叉驗證，將數據分為4份
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
    return tr_x, tr_y, va_x, va_y

def create_model(params):
    tr_x, tr_y, va_x, va_y = prepare_data()  # 確保數據被準備好
    
    # 創建一個Sequential模型
    model = Sequential()

    # 添加卷積層
    model.add(Conv2D(filters=params['conv1_filters'],               # 卷積層的濾波器數量
                     kernel_size=params['conv1_kernel_size'],       # 卷積核的大小
                     padding='same',                                # 使用same填充以保持輸出尺寸
                     activation=params['conv1_activation'],         # 卷積層的激活函數
                     input_shape=(28, 28, 1)                        # 輸入數據的形狀
                     ))
    
    model.add(Conv2D(filters=params['conv2_filters'],               # 添加第二個卷積層
                     kernel_size=params['conv2_kernel_size'],       # 卷積核的大小
                     padding='same',                                # 使用same填充以保持輸出尺寸
                     activation=params['conv2_activation']          # 卷積層的激活函數
                     ))
    model.add(MaxPooling2D(pool_size=(2, 2)))                       # 添加最大池化層，池化窗口為2x2

    model.add(Dropout(params['dropout1']))                          # 添加Dropout層以防止過擬合，丟棄比例由超參數指定

    model.add(Conv2D(filters=params['conv3_filters'],               # 添加第三個卷積層
                     kernel_size=params['conv3_kernel_size'],       # 卷積核的大小
                     padding='same',                                # 使用same填充以保持輸出尺寸
                     activation='relu'                              # 卷積層的激活函數為relu
                     ))                     
                     
    model.add(Conv2D(filters=params['conv4_filters'],               # 添加第四個卷積層
                     kernel_size=params['conv4_kernel_size'],       # 卷積核的大小
                     padding='same',                                # 使用same填充以保持輸出尺寸
                     activation=params['conv4_activation']          # 卷積層的激活函數
                     ))                     
    model.add(MaxPooling2D(pool_size=(2, 2)))   # 添加第二個最大池化層，池化窗口為2x2

    model.add(Dropout(params['dropout2']))      # 添加Dropout層以防止過擬合，丟棄比例由超參數指定
    model.add(Flatten())                        # 將多維輸出展平成一維數據

    # 添加全連接層
    if params['fc_layers'] == 'one':                                # 根據選擇添加一個全連接層
        model.add(Dense(params['fc1_units'],                        # 全連接層，具有由超參數指定的神經元數量
                        activation=params['fc1_activation']))       # 激活函數
        model.add(Dropout(params['dropout_fc1']))                   # 添加Dropout層以防止過擬合，丟棄比例由超參數指定

    elif params['fc_layers'] == 'two':                              # 根據選擇添加兩個全連接層
        model.add(Dense(params['fc1_units'],                        # 第一個全連接層，具有由超參數指定的神經元數量
                        activation=params['fc1_activation']))       # 激活函數
        model.add(Dropout(params['dropout_fc1']))                   # 添加Dropout層以防止過擬合，丟棄比例由超參數指定
             
        model.add(Dense(params['fc2_units'],                        # 第二個全連接層，具有由超參數指定的神經元數量
                        activation=params['fc2_activation']))       # 激活函數
        model.add(Dropout(params['dropout_fc2']))                   # 添加Dropout層以防止過擬合，丟棄比例由超參數指定

    model.add(Dense(10, activation="softmax"))                      # 最終輸出層，具有10個神經元和softmax激活函數（用於分類）

    # 編譯模型
    model.compile(loss="categorical_crossentropy",              # 設置損失函數為categorical_crossentropy
                  optimizer=params['optimizer'],                # 設置優化器
                  metrics=["accuracy"])                         # 設置評估指標為準確率

    epoch = 30  # 設置訓練輪數為30
    batch_size = params['batch_size']  # 設置批次大小
    
    # 訓練模型
    result = model.fit(tr_x, tr_y,                              # 使用訓練數據進行模型訓練
                       epochs=epoch,                            # 訓練的總輪數
                       batch_size=batch_size,                   # 批次大小
                       validation_data=(va_x, va_y),            # 設置驗證數據
                       verbose=0)                               # 設置訓練過程的輸出詳情

    validation_acc = np.amax(result.history['val_accuracy'])      # 獲取最佳驗證準確率
    print('Best validation acc of epoch:', validation_acc)        # 打印最佳驗證準確率

    # 返回模型的結果，包括損失值（取負值以便Hyperopt進行最大化）和狀態
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

# 定義超參數空間
space = {
    # 卷積層1的濾波器數量，選擇32或64
    'conv1_filters': hp.choice('conv1_filters', [32, 64]),

    # 卷積層1的卷積核大小，選擇(3, 3)、(5, 5)或(7, 7)
    'conv1_kernel_size': hp.choice('conv1_kernel_size', [(3, 3), (5, 5), (7, 7)]),

    # 卷積層1的激活函數，選擇'tanh'或'relu'
    'conv1_activation': hp.choice('conv1_activation', ['tanh', 'relu']),
    
    # 卷積層2的濾波器數量，選擇32或64
    'conv2_filters': hp.choice('conv2_filters', [32, 64]),

    # 卷積層2的卷積核大小，選擇(3, 3)、(5, 5)或(7, 7)
    'conv2_kernel_size': hp.choice('conv2_kernel_size', [(3, 3), (5, 5), (7, 7)]),

    # 卷積層2的激活函數，選擇'tanh'或'relu'
    'conv2_activation': hp.choice('conv2_activation', ['tanh', 'relu']),
    
    # 卷積層3的濾波器數量，選擇32或64
    'conv3_filters': hp.choice('conv3_filters', [32, 64]),

    # 卷積層3的卷積核大小，選擇(3, 3)、(5, 5)或(7, 7)
    'conv3_kernel_size': hp.choice('conv3_kernel_size', [(3, 3), (5, 5), (7, 7)]),
    
    # 卷積層4的濾波器數量，選擇32或64
    'conv4_filters': hp.choice('conv4_filters', [32, 64]),

    # 卷積層4的卷積核大小，選擇(3, 3)、(5, 5)或(7, 7)
    'conv4_kernel_size': hp.choice('conv4_kernel_size', [(3, 3), (5, 5), (7, 7)]),

    # 卷積層4的激活函數，選擇'tanh'或'relu'
    'conv4_activation': hp.choice('conv4_activation', ['tanh', 'relu']),
    
    # 卷積層1後的Dropout比例，範圍在0.2到0.6之間
    'dropout1': hp.uniform('dropout1', 0.2, 0.6),

    # 卷積層2後的Dropout比例，範圍在0.2到0.6之間
    'dropout2': hp.uniform('dropout2', 0.2, 0.6),
    
    # 全連接層的層數，選擇'one'或'two'
    'fc_layers': hp.choice('fc_layers', ['one', 'two']),

    # 第一個全連接層的單元數量，選擇500、600或700
    'fc1_units': hp.choice('fc1_units', [500, 600, 700]),

    # 第一個全連接層的激活函數，選擇'tanh'或'relu'
    'fc1_activation': hp.choice('fc1_activation', ['tanh', 'relu']),

    # 第一個全連接層後的Dropout比例，範圍在0.1到0.6之間
    'dropout_fc1': hp.uniform('dropout_fc1', 0.1, 0.6),

    # 第二個全連接層的單元數量，選擇100、150或200（僅在fc_layers為'two'時使用）
    'fc2_units': hp.choice('fc2_units', [100, 150, 200]),

    # 第二個全連接層的激活函數，選擇'tanh'或'relu'（僅在fc_layers為'two'時使用）
    'fc2_activation': hp.choice('fc2_activation', ['tanh', 'relu']),

    # 第二個全連接層後的Dropout比例，範圍在0.2到0.6之間（僅在fc_layers為'two'時使用）
    'dropout_fc2': hp.uniform('dropout_fc2', 0.2, 0.6),
    
    # 優化器，選擇'adam'或'rmsprop'
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),

    # 批次大小，選擇100、200或300
    'batch_size': hp.choice('batch_size', [100, 200, 300])
}

trials = Trials()  # 初始化 Trials 物件

# 使用fmin進行超參數優化
best_run = fmin(fn = create_model,                         # 設置要優化的模型函數
                space = space,                             # 設置超參數空間
                algo = tpe.suggest,                        # 設置優化演算法為TPE
                max_evals = 75,                            # 設置最大評估次數為75
                trials = trials)                           # 設置實驗記錄
print(best_run)                                            # 打印最佳參數組合

# 使用最佳模型進行評估
_, tr_y, va_x, va_y = prepare_data()                    # 重新準備數據
best_model = trials.best_trial['result']['model']       # 獲取最佳模型
val_loss, val_acc = best_model.evaluate(va_x, va_y)     # 評估最佳模型在驗證數據上的損失和準確率
print("val_loss : ", val_loss)                          # 打印驗證損失
print("val_acc : ", val_acc)                            # 打印驗證準確率
