import numpy as np                  # 匯入 numpy 庫，用於數值計算
import tensorflow as tf             # 匯入 TensorFlow 庫，用於深度學習
import pandas as pd                 # 匯入 pandas 庫，用於資料操作
import matplotlib.pyplot as plt     # 匯入 matplotlib 庫，用於繪圖
import h5py                         # 匯入 h5py 庫，用於讀寫 HDF5 格式文件

# 定義損失函數
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))  # 使用均方誤差作為損失函數

w = 0  # 初始化權重 w 為 0
b = 0  # 初始化偏差 b 為 0

# 定義預測函數
def predict(X):
    return w * X + b  # 線性回歸模型的預測公式 y = wx + b

# 定義訓練函數
def train(X, y, epochs=10000, lr=0.001):
    current_loss = 0                                    # 初始化當前損失為 0
    for epoch in range(epochs):                         # 迭代指定的次數
        with tf.GradientTape() as t:                    # 使用 TensorFlow 的 GradientTape 來自動計算梯度
            t.watch(tf.constant(X))                     # 監視輸入 X
            current_loss = loss(y, predict(X))          # 計算當前損失

        dw, db = t.gradient(current_loss, [w, b])               # 計算損失對 w 和 b 的梯度
        w.assign_sub(lr * dw)                                   # 使用梯度和學習率更新 w
        b.assign_sub(lr * db)                                   # 使用梯度和學習率更新 b
        print(f'Epoch {epoch}: Loss: {current_loss.numpy()}')   # 輸出每次迭代的損失

# 從 CSV 文件讀取資料
data = pd.read_csv("./Salary_Data.csv")

# 提取特徵和標籤
X = tf.constant(data["YearsExperience"], dtype=tf.float32)  # 將工作經驗年限轉換為 TensorFlow 常量
y = tf.constant(data["Salary"], dtype=tf.float32)           # 將薪水轉換為 TensorFlow 常量

# w、b 初始值均設為 0
w = tf.Variable(0.0)  # 將 w 設置為 TensorFlow 變量，初始值為 0
b = tf.Variable(0.0)  # 將 b 設置為 TensorFlow 變量，初始值為 0

# 執行訓練
train(X, y)  # 調用訓練函數

# w、b 的最佳解
print(f'w = {w.numpy()}, b = {b.numpy()}')          # 輸出最終的 w 和 b 值

plt.scatter(X, y, label='data')                     # 繪製資料點
plt.plot(X, predict(X), 'r-', label='predicted')    # 繪製回歸線
plt.legend()                                        # 顯示圖例
plt.show()                                          # 顯示圖表

# 保存模型到 .h5 格式
model = tf.keras.Sequential([                   # 建立 Keras 順序模型
    tf.keras.layers.Dense(1, input_shape=(1,))  # 添加一個全連接層，輸入形狀為單一特徵
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='mean_squared_error')    # 編譯模型，使用 SGD 優化器和均方誤差損失函數
model.layers[0].set_weights([np.array([[w.numpy()]]), np.array([b.numpy()])])                       # 設置模型權重為訓練後的 w 和 b

# 保存模型到 .h5 文件
model.save('my_LR_model.h5')         
