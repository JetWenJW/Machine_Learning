from hyperopt import hp, Trials, tpe, fmin, STATUS_OK                   # 從 hyperopt 套件中引入所需的模組和函數
import numpy as np                                                      # 引入 numpy 模組
import pandas as pd                                                     # 引入 pandas 模組
from sklearn.model_selection import KFold                               # 引入 KFold 交叉驗證模組
from tensorflow.keras.utils import to_categorical                       # 引入 to_categorical 函數，用於進行 one-hot 編碼
from tensorflow.keras.models import Sequential                          # 引入 Sequential 模型
from tensorflow.keras.layers import Dense, Activation, Dropout, Input   # 引入所需的層
from tensorflow.keras.optimizers import Adam, RMSprop                   # 引入所需的優化器


def prepare_data():
    """
    準備數據函數，讀取訓練數據並進行預處理
    """
    train = pd.read_csv('./digit-recognizer/train.csv')     # 讀取訓練數據
    train_x = train.drop(['label'], axis=1)                 # 刪除標籤列，僅保留特徵
    train_y = train['label']                                # 提取標籤

    kf = KFold(n_splits=4, shuffle=True, random_state=123)          # 進行 4 折交叉驗證(總共分成4組)
    tr_idx, va_idx = list(kf.split(train_x))[0]                     # 獲取訓練和驗證數據索引(取第0號組)
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]         # 根據索引分割訓練和驗證數據
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]         # 根據索引分割訓練和驗證標籤
    tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)     # 將特徵標準化至 [0, 1] 區間

    tr_y = to_categorical(tr_y, 10)  # 將訓練標籤進行 one-hot 編碼
    va_y = to_categorical(va_y, 10)  # 將驗證標籤進行 one-hot 編碼

    return tr_x, tr_y, va_x, va_y  # 返回處理後的數據

def create_model(params):
    """
    創建模型函數，根據不同參數構建模型
    """
    tr_x, tr_y, va_x, va_y = prepare_data()  # 準備數據
    
    model = Sequential()                                    # 初始化模型
    model.add(Input(shape=(784,)))                          # 使用 Input 層定義輸入形狀
    model.add(Dense(params['units'], activation='relu'))    # 添加隱藏層，單元數由參數決定
    model.add(Dropout(0.4))                                 # 添加 Dropout 層，丟棄率為 0.4

    if params['layers'] == 'one':
        model.add(Dense(params['units_2'], activation='relu'))  # 根據參數添加第二層
    elif params['layers'] == 'two':
        model.add(Dense(params['units_2'], activation='relu'))  # 根據參數添加第二層
        model.add(Dense(params['units_3'], activation='relu'))  # 根據參數添加第三層

    model.add(Dense(10, activation='softmax'))                                                  # 添加輸出層，使用 softmax 激活函數
    optimizer = Adam() if params['optimizer'] == 'adam' else RMSprop()                          # 根據參數選擇優化器
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])   # 編譯模型
    
    result = model.fit(tr_x, tr_y, epochs=5, batch_size=100, validation_data=(va_x, va_y), verbose=0)   # 訓練模型
    validation_acc = np.amax(result.history['val_accuracy'])                                            # 獲取驗證集上的最高準確率
    print('Accuracy in search:', validation_acc)                                                        # 輸出驗證準確率

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}                               # 返回結果

# 定義超參數搜索空間
space = {
    'units': hp.choice('units', [500, 784]),                    # 隱藏層單元數選項
    'layers': hp.choice('layers', ['none', 'one', 'two']),      # 隱藏層層數選項
    'units_2': hp.choice('units_2', [100, 200]),                # 第二隱藏層單元數選項
    'units_3': hp.choice('units_3', [25, 50]),                  # 第三隱藏層單元數選項
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop'])    # 優化器選項
}

trials = Trials()  # 初始化 Trials 物件
best_run = fmin(
    create_model,         # 要優化的目標函數，這裡是創建和評估模型的函數
    space,                # 定義的超參數搜索空間，包括所有可能的超參數選項
    algo = tpe.suggest,   # 使用TPE（Tree-structured Parzen Estimator）算法進行優化
    max_evals = 20,       # 最大評估次數，即進行20次不同的超參數組合嘗試
    trials = trials       # 初始化的Trials物件，用來記錄和管理每次嘗試的結果
)  # 進行超參數優化

print("Best Parameter : ", best_run)  # 輸出最佳參數

# 獲取最佳模型
best_model = None           # 初始化最佳模型變數為 None，表示尚未找到最佳模型
best_loss = float('inf')    # 設定初始最佳損失為無窮大，這樣所有找到的損失都會比它小

# 遍歷 Trials 物件中的每一個 trial（即每次超參數試驗的結果）
for trial in trials.trials:
    # 檢查 trial 中是否存在 'result' 鍵，且 'result' 中是否有 'loss' 鍵，
    # 並且 'result' 的狀態為 STATUS_OK（表示試驗完成且結果有效）
    if 'result' in trial and 'loss' in trial['result'] and trial['result']['status'] == STATUS_OK:
        # 獲取當前 trial 的損失值
        current_loss = trial['result']['loss']
        
        # 如果當前損失值比已知的最佳損失值還要小，則更新最佳損失和最佳模型
        if current_loss < best_loss:
            best_loss = current_loss  # 更新最佳損失值
            best_model = trial['result']['model']  # 更新最佳模型
        # 找到最佳模型後退出循環，因為只需要找到一個最佳模型

_, _, va_x, va_y = prepare_data()                       # 準備驗證數據
val_loss, val_acc = best_model.evaluate(va_x, va_y)     # 評估最佳模型在驗證集上的表現
print("val_loss:", val_loss)                            # 輸出驗證集上的損失
print("val_acc:", val_acc)                              # 輸出驗證集上的準確率
