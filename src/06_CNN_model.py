import numpy as np                                  # 載入數據處理套件
import pandas as pd                                 # 載入數據處理套件
from tensorflow.keras.utils import to_categorical   # 載入將標籤轉為one-hot編碼的函數
from sklearn.model_selection import KFold           # 載入KFold交叉驗證工具

# 讀取訓練資料集
train = pd.read_csv('./digit-recognizer/train.csv')
# 分離特徵和標籤
train_x = train.drop(['label'], axis=1)
train_y = train['label']

# 讀取測試資料集
test_x = pd.read_csv('./digit-recognizer/test.csv')

# 使用KFold進行交叉驗證，分為4份，打亂數據，隨機種子為71
kf = KFold(n_splits = 4, shuffle = True, random_state = 71)
# 獲取第一組訓練和驗證索引
tr_idx, va_idx = list(kf.split(train_x))[0]
# 根據索引分割訓練和驗證數據
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# 將特徵數據標準化到0-1之間
tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)

# 將特徵數據重塑為28x28x1的形狀，以適應CNN模型
tr_x = tr_x.reshape(-1, 28, 28, 1)
va_x = va_x.reshape(-1, 28, 28, 1)

# 將標籤數據轉為one-hot編碼
tr_y = to_categorical(tr_y, 10)
va_y = to_categorical(va_y, 10)

# 打印訓練和驗證數據的形狀
print(tr_x.shape)
print(tr_y.shape)
print(va_x.shape)
print(va_y.shape)

# 載入Keras的神經網路模型構建工具
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# 建立一個序列模型
model = Sequential()

# 添加一個卷積層，這個層會對輸入數據進行卷積操作
model.add(Conv2D(
    filters = 32,                  # 設置濾波器的數量為32，即該層會學習32個不同的濾波器
    kernel_size = (5, 5),          # 設置每個濾波器的大小為5x5像素
    padding = 'same',              # 使用 'same' 填充模式，確保輸出尺寸與輸入尺寸相同（保持邊界像素）
    input_shape = (28, 28, 1),     # 定義輸入數據的形狀為28x28像素，1表示單通道（灰階圖像）
    activation = 'relu'            # 使用 ReLU（Rectified Linear Unit）激活函數，對每個輸出值進行非線性變換
))

# 添加一個平坦層，將多維數據展平成一維
model.add(Flatten())

# 添加一個全連接層，包含10個神經元，使用softmax激活函數
model.add(Dense(10,
                activation = 'softmax'))

# 設置模型的損失函數為categorical_crossentropy，優化器為rmsprop，評估指標為準確度
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

# 顯示模型摘要
model.summary()

# 訓練模型，使用訓練資料進行20個訓練週期，批次大小為100，驗證資料為驗證集
history = model.fit(tr_x, tr_y,
                    epochs = 20,                          # 訓練20個週期
                    batch_size = 100,                     # 批次大小為100
                    verbose = 1,                          # 顯示訓練過程中的訊息
                    validation_data = (va_x, va_y))       # 使用驗證集進行驗證

# 繪製訓練和驗證的損失函數變化圖
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6))
plt.subplots_adjust(wspace=0.2)

# 繪製損失函數變化圖，圖像在第1行第2列的位置
plt.subplot(1, 2, 1)

# 繪製訓練損失（training loss）隨著訓練週期（epoch）的變化，使用虛線表示
plt.plot(history.history['loss'],
         linestyle='--',                    # 使用虛線
         label='training')                  # 標籤為 'training'

# 繪製驗證損失（validation loss）隨著訓練週期（epoch）的變化
plt.plot(history.history['val_loss'],
         label='validation')        # 標籤為 'validation'
plt.ylim(0, 1)                      # 設定y軸的顯示範圍為0到1
plt.legend()                        # 顯示圖例，用於標識不同曲線的含義
plt.grid()                          # 顯示網格線，以便更容易閱讀圖表
plt.xlabel('epoch')                 # 設定x軸的標籤為 'epoch'，表示訓練的輪數
plt.ylabel('loss')                  # 設定y軸的標籤為 'loss'，表示損失函數值



# 繪製準確度變化圖，圖像在第1行第2列的位置
plt.subplot(1, 2, 2)

# 繪製訓練準確度（training accuracy）隨著訓練週期（epoch）的變化，使用虛線表示
plt.plot(history.history['accuracy'],
         linestyle='--',  # 使用虛線
         label='training')  # 標籤為 'training'

# 繪製驗證準確度（validation accuracy）隨著訓練週期（epoch）的變化
plt.plot(history.history['val_accuracy'],
         label='validation')        # 標籤為 'validation'
plt.ylim(0.5, 1)                    # 設定y軸的顯示範圍為0.5到1
plt.legend()                        # 顯示圖例，用於標識不同曲線的含義
plt.grid()                          # 顯示網格線，以便更容易閱讀圖表
plt.xlabel('epoch')                 # 設定x軸的標籤為 'epoch'，表示訓練的輪數
plt.ylabel('acc')                   # 設定y軸的標籤為 'acc'，表示準確度（accuracy）



# 顯示圖表
plt.show()
