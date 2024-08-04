import matplotlib.pyplot as plt                                     # 引入 matplotlib.pyplot 用於繪製圖像
import numpy as np                                                  # 引入 numpy 用於數值運算
import pandas as pd                                                 # 引入 pandas 用於數據處理
from tensorflow.keras.utils import to_categorical                   # 引入 to_categorical 函數，用於進行 one-hot 編碼
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 引入 ImageDataGenerator 用於數據增強

def draw(X):
    """
    顯示一批圖像
    """
    plt.figure(figsize=(8, 8))          # 設置圖像大小
    pos = 1                             # 初始化圖像位置

    for i in range(X.shape[0]):         # 遍歷圖像批次
        plt.subplot(4, 4, pos)          # 創建 4x4 子圖
        plt.imshow(X[i].reshape((28, 28)), interpolation='nearest')  # 顯示圖像
        plt.axis('off')     # 關閉坐標軸
        pos += 1            # 位置遞增
    plt.show()              # 顯示圖像

# 讀取資料
train = pd.read_csv('./digit-recognizer/train.csv')     # 讀取訓練數據
tr_x = train.drop(['label'], axis=1)                    # 刪除標籤列，僅保留特徵
train_y = train['label']                                # 提取標籤

# 資料前處理
tr_x = np.array(tr_x / 255.0)           # 將特徵標準化至 [0, 1] 區間
tr_x = tr_x.reshape(-1, 28, 28, 1)      # 重塑特徵為 28x28x1 的格式
tr_y = to_categorical(train_y, 10)      # 將標籤進行 one-hot 編碼
batch_size = 16                         # 設置批次大小

# 顯示原始圖像
draw(tr_x[0:batch_size])                # 顯示前 16 張圖像

# 進行數據增強並顯示結果

# Rotation Shift (旋轉變換)
datagen = ImageDataGenerator(rotation_range=90)                     # 設置旋轉範圍
g = datagen.flow(tr_x, tr_y, batch_size=batch_size, shuffle=False)  # 創建數據生成器
for X_batch, y_batch in g:
    draw(X_batch)                                                   # 顯示增強後的圖像
    break                                                           # 只顯示第一個批次

# Width Shift (水平平移)
datagen = ImageDataGenerator(width_shift_range=0.5)                 # 設置水平平移範圍
g = datagen.flow(tr_x, tr_y, batch_size=batch_size, shuffle=False)  # 創建數據生成器
for X_batch, y_batch in g:
    draw(X_batch)                                                   # 顯示增強後的圖像
    break                                                           # 只顯示第一個批次

# Height Shift (垂直平移)
datagen = ImageDataGenerator(height_shift_range=0.5)                # 設置垂直平移範圍
g = datagen.flow(tr_x, tr_y, batch_size=batch_size, shuffle=False)  # 創建數據生成器
for X_batch, y_batch in g:
    draw(X_batch)                                                   # 顯示增強後的圖像
    break                                                           # 只顯示第一個批次

# Zoom (縮放)
datagen = ImageDataGenerator(zoom_range=0.5)                        # 設置縮放範圍
g = datagen.flow(tr_x, tr_y, batch_size=batch_size, shuffle=False)  # 創建數據生成器
for X_batch, y_batch in g:
    draw(X_batch)                                                   # 顯示增強後的圖像
    break                                                           # 只顯示第一個批次

# Horizontal Flip (水平翻轉)
datagen = ImageDataGenerator(horizontal_flip=True)                  # 設置水平翻轉
g = datagen.flow(tr_x, tr_y, batch_size=batch_size, shuffle=False)  # 創建數據生成器
for X_batch, y_batch in g:
    draw(X_batch)                                                   # 顯示增強後的圖像
    break                                                           # 只顯示第一個批次

# Vertical Flip (垂直翻轉)
datagen = ImageDataGenerator(vertical_flip=True)                    # 設置垂直翻轉
g = datagen.flow(tr_x, tr_y, batch_size=batch_size, shuffle=False)  # 創建數據生成器
for X_batch, y_batch in g:
    draw(X_batch)                                                   # 顯示增強後的圖像
    break                                                           # 只顯示第一個批次
