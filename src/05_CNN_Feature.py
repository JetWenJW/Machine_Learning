import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取數據
train = pd.read_csv('./digit-recognizer/train.csv')     # 從CSV檔案讀取數據
train_x = train.drop(['label'], axis=1)                 # 移除標籤列，保留特徵數據

# 數據標準化
train_x = np.array(train_x / 255.0)                     # 將像素值標準化到0-1範圍

# 重塑數據形狀
tr_x = train_x.reshape(-1, 28, 28, 1)                   # 將數據重塑為28x28的圖像，1代表單通道（灰階）

# 定義邊緣檢測過濾器
vertical_edge_fil = np.array(
    [[-2, 1, 1],   # 垂直邊緣檢測過濾器
     [-2, 1, 1],
     [-2, 1, 1]]
)

horizontal_edge_fil = np.array(
    [[1, 1, 1],    # 水平邊緣檢測過濾器
     [1, 1, 1],
     [-2, -2, -2]], 
    dtype=float
)

# 設定圖像ID
img_id = 21                     # 選擇要處理的圖像ID
img_x = tr_x[img_id, :, :, 0]   # 提取指定圖像的數據
img_height = 28                 # 圖像高度
img_width = 28                  # 圖像寬度
img_x = img_x.reshape(img_height, img_width)  # 重塑為28x28的矩陣

# 初始化邊緣圖像
vertical_edge = np.zeros_like(img_x)    # 初始化垂直邊緣圖像，大小與原始圖像相同
horizontal_edge = np.zeros_like(img_x)  # 初始化水平邊緣圖像，大小與原始圖像相同

# 邊緣檢測操作
for h in range(img_height - 3):                 # 遍歷每個可能的起始點（高度）
    for w in range(img_width - 3):              # 遍歷每個可能的起始點（寬度）
        img_region = img_x[h:h + 3, w:w + 3]    # 提取3x3的區域
        # 計算垂直邊緣
        vertical_edge[h + 1, w + 1] = np.dot(
            img_region.reshape(-1),             # 將區域展平為一維數組
            vertical_edge_fil.reshape(-1)       # 將過濾器展平為一維數組
        )
        # 計算水平邊緣
        horizontal_edge[h + 1, w + 1] = np.dot(
            img_region.reshape(-1),             # 將區域展平為一維數組
            horizontal_edge_fil.reshape(-1)     # 將過濾器展平為一維數組
        )

# 繪製結果
plt.figure(figsize=(8, 8))          # 設定畫布大小
plt.subplots_adjust(wspace=0.2)     # 調整子圖之間的間距
plt.gray()                          # 設定灰度模式

# 顯示原始圖像
plt.subplot(2, 2, 1)                # 設定子圖位置
plt.pcolor(1 - img_x)               # 顯示圖像，將像素值反轉以提高對比度
plt.xlim(-1, 29)                    # 設定x軸範圍
plt.ylim(29, -1)                    # 設定y軸範圍，y軸反向顯示

# 顯示垂直邊緣檢測結果
plt.subplot(2, 2, 3)                # 設定子圖位置
plt.pcolor(-vertical_edge)          # 顯示垂直邊緣檢測結果，將像素值反轉以提高對比度
plt.xlim(-1, 29)                    # 設定x軸範圍
plt.ylim(29, -1)                    # 設定y軸範圍，y軸反向顯示

# 顯示水平邊緣檢測結果
plt.subplot(2, 2, 4)                # 設定子圖位置
plt.pcolor(-horizontal_edge)        # 顯示水平邊緣檢測結果，將像素值反轉以提高對比度
plt.xlim(-1, 29)                    # 設定x軸範圍
plt.ylim(29, -1)                    # 設定y軸範圍，y軸反向顯示

plt.show()                          # 顯示所有繪圖
