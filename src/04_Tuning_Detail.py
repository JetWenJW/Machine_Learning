from hyperopt import fmin, tpe, hp, STATUS_OK, Trials                   # 從hyperopt庫中導入所需的函數和模組
import numpy as np                                                      # 導入NumPy庫，用於數學運算
import pandas as pd                                                     # 導入Pandas庫，用於數據處理
from sklearn.model_selection import KFold                               # 從scikit-learn導入KFold交叉驗證
from tensorflow.keras.utils import to_categorical                       # 導入to_categorical，用於one-hot編碼
from tensorflow.keras.models import Sequential                          # 導入Sequential模型
from tensorflow.keras.layers import Dense, Activation, Dropout, Input   # 導入各種層
from tensorflow.keras.optimizers import RMSprop                         # 導入RMSprop優化器

def prepare_data():
    # 讀取數據
    train = pd.read_csv('./digit-recognizer/train.csv')     # 讀取訓練數據
    train_x = train.drop(['label'], axis=1)                 # 去除標籤列，保留特徵數據
    train_y = train['label']                                # 提取標籤列
    test_x = pd.read_csv('./digit-recognizer/test.csv')     # 讀取測試數據（不使用）

    # 設置KFold交叉驗證
    kf = KFold(n_splits=4, shuffle=True, random_state=123)      # 4折交叉驗證
    tr_idx, va_idx = list(kf.split(train_x))[0]                 # 獲取第一折的訓練和驗證索引
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]     # 根據索引分割數據
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]     # 根據索引分割標籤

    # 數據標準化
    tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)  # 將像素值標準化到0-1範圍

    # 將標籤進行one-hot編碼
    tr_y = to_categorical(tr_y, 10)  # 將訓練標籤轉換為one-hot編碼
    va_y = to_categorical(va_y, 10)  # 將驗證標籤轉換為one-hot編碼

    return tr_x, tr_y, va_x, va_y       # 返回處理後的數據

def create_model(params):
    # 準備數據
    tr_x, tr_y, va_x, va_y = prepare_data()  # 調用prepare_data函數

    # 建立模型
    model = Sequential()                                        # 創建Sequential模型
    model.add(Input(shape=(784,)))                              # 添加輸入層，輸入形狀為784（28x28圖像展平）
    model.add(Dense(784, activation=params['activation']))      # 添加第一個全連接層，激活函數由params決定
    model.add(Dropout(params['dropout']))                       # 添加Dropout層，丟棄率由params決定
    model.add(Dense(200, activation=params['activation']))      # 添加第二個全連接層，激活函數由params決定
    model.add(Dropout(params['dropout']))                       # 添加Dropout層，丟棄率由params決定
    model.add(Dense(25, activation=params['activation']))       # 添加第三個全連接層，激活函數由params決定
    model.add(Dropout(params['dropout']))                       # 添加Dropout層，丟棄率由params決定
    model.add(Dense(10, activation='softmax'))                  # 添加輸出層，10個類別，使用softmax激活函數

    # 編譯模型
    model.compile(loss='categorical_crossentropy',  # 損失函數為categorical_crossentropy
                  optimizer=RMSprop(),              # 優化器為RMSprop
                  metrics=['accuracy'])             # 評估指標為準確率

    # 訓練模型
    result = model.fit(tr_x, tr_y,                          # 訓練數據和標籤
                       epochs=20,                           # 訓練20個時代
                       batch_size=params['batch_size'],     # 批次大小由params決定
                       validation_data=(va_x, va_y),        # 驗證數據和標籤
                       verbose=0)                           # 訓練過程不輸出詳細信息

    # 取得最佳驗證準確率
    validation_acc = np.amax(result.history['val_accuracy'])                # 獲取驗證準確率中的最大值
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}   # 返回結果，loss為負數以便最小化

# 定義超參數空間
space = {
    'activation': hp.choice('activation', ['tanh', 'relu']),    # 激活函數選擇
    'dropout': hp.uniform('dropout', 0.2, 0.4),                 # Dropout丟棄率選擇
    'batch_size': hp.choice('batch_size', [100, 200])           # 批次大小選擇
}

# 初始化Trials對象
trials = Trials()

# 使用Hyperopt進行超參數優化
best_run = fmin(create_model,       # 優化的目標函數
                space,              # 超參數空間
                algo=tpe.suggest,   # 使用TPE算法進行優化
                max_evals=100,      # 最多嘗試100次
                trials=trials)      # 保存結果的Trials對象

print(best_run)                     # 輸出最佳參數設置

# 獲取最佳模型
best_model = None
for trial in trials.trials:  # 遍歷所有試驗結果
    if trial['result']['status'] == STATUS_OK and trial['result']['loss'] == -best_run['loss']:
        best_model = trial['result']['model']  # 找到對應的最佳模型
        break

# 評估最佳模型
_, _, va_x, va_y = prepare_data()  # 重新準備數據
val_loss, val_acc = best_model.evaluate(va_x, va_y)  # 評估模型性能
print("val_loss:", val_loss)    # 輸出驗證損失
print("val_acc:", val_acc)      # 輸出驗證準確率
