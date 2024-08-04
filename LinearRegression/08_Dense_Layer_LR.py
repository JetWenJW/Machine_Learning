# 載入所需套件
import numpy as np                  # 匯入 numpy，用於數值計算
import tensorflow as tf             # 匯入 TensorFlow，用於深度學習
import pandas as pd                 # 匯入 pandas，用於資料處理
import matplotlib.pyplot as plt     # 匯入 matplotlib，用於資料視覺化

# 從 CSV 檔案中讀取資料
data = pd.read_csv("./Salary_Data.csv")

# y = w*x + b
# 提取特徵和標籤
X = tf.constant(data["YearsExperience"], dtype=tf.float32)  # 將工作經驗年限轉換為 TensorFlow 常量
y = tf.constant(data["Salary"], dtype=tf.float32)           # 將薪水轉換為 TensorFlow 常量

# w、b 初始值均設為 0
w = tf.Variable(0.0)
b = tf.Variable(0.0)

# 定義完全連接層(Dense)
# units：輸出神經元個數，input_shape：輸入神經元個數
layer1 = tf.keras.layers.Dense(units=1, input_shape=[1]) 

# 定義神經網路模型，包含一層完全連接層
model = tf.keras.Sequential([layer1])

# 定義模型的損失函數(loss)為均方誤差(MSE)，優化器(optimizer)為隨機梯度下降(SGD)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='mean_squared_error'
)

# 設定完全連接層的初始權重和偏差
model.layers[0].set_weights([np.array([[w.numpy()]]), np.array([b.numpy()])])

# 訓練模型，訓練週期為 10000，verbose=True 表示顯示訓練過程中的訊息
history = model.fit(X, y, epochs=10000, verbose=True)

# 保存訓練好的模型到 HDF5 檔案
model.save('Dense_LR_model.h5')
print("-" * 10, "Save Model Done ~", "-" * 10)

# 繪製訓練過程中的損失函數變化圖
plt.xlabel('訓練週期', fontsize=20)
plt.ylabel("損失函數(loss)", fontsize=20)
plt.plot(history.history['loss'])

# 獲取訓練後的權重和偏差
w = layer1.get_weights()[0][0][0]
b = layer1.get_weights()[1][0]
print(f"w：{w:.4f} , b：{b:.4f}")

# 繪製資料點和預測線
plt.scatter(X, y, label='data')
plt.plot(X, X * w + b, 'r-', label='predicted')
plt.legend()
plt.show()
