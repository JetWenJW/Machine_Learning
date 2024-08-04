import numpy as np                              # 導入NumPy庫，用於數學運算
from tensorflow.keras.datasets import cifar10   # 從TensorFlow Keras導入CIFAR-10數據集

# 下載並載入CIFAR-10數據集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()  # 分別獲取訓練和測試數據及標籤

# 打印數據的形狀
print('X_train: ', X_train.shape, 'y_train: ', y_train.shape)   # 輸出訓練數據和標籤的形狀
print('X_test: ', X_test.shape, 'y_test: ', y_test.shape)       # 輸出測試數據和標籤的形狀

import matplotlib.pyplot as plt                 # 導入Matplotlib庫，用於數據可視化

num_classes = 10    # CIFAR-10數據集有10個類別
pos = 1             # 圖片在子圖中的位置，從1開始

# 遍歷每一個類別
for target_class in range(num_classes):
    target_idx = []  # 用於存儲當前類別的索引列表

    # 遍歷所有訓練數據，找到屬於當前類別的圖片
    for i in range(len(y_train)):
        if y_train[i][0] == target_class:   # 如果標籤等於當前類別
            target_idx.append(i)            # 將索引添加到列表中
    
    np.random.shuffle(target_idx)           # 隨機打亂索引列表
    plt.figure(figsize = (20, 20))          # 創建一個20x20英寸的圖形
    
    # 顯示當前類別的前10張圖片
    for idx in target_idx[:10]:             # 只選擇前10張圖片
        plt.subplot(10, 10, pos)            # 創建10x10的子圖，指定位置
        plt.imshow(X_train[idx])            # 顯示當前圖片
        pos += 1                            # 更新位置索引，準備顯示下一張圖片

plt.show()  # 顯示所有圖形
