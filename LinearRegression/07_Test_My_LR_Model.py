import numpy as np                  # 匯入 numpy 庫，用於數值計算
import tensorflow as tf             # 匯入 TensorFlow 庫，用於深度學習
import pandas as pd                 # 匯入 pandas 庫，用於資料操作
import matplotlib.pyplot as plt     # 匯入 matplotlib 庫，用於繪圖

# 從 CSV 文件讀取資料
data = pd.read_csv("./Salary_Data.csv")

# 提取特徵和標籤
X = tf.constant(data["YearsExperience"], dtype=tf.float32)  # 將工作經驗年限轉換為 TensorFlow 常量
y = tf.constant(data["Salary"], dtype=tf.float32)           # 將薪水轉換為 TensorFlow 常量

# 加載模型
loaded_model = tf.keras.models.load_model('Dense_LR_model.h5')  # 從 HDF5 文件加載已保存的模型

# 預測
X_test = tf.constant(data["YearsExperience"], dtype=tf.float32)     # 測試集（工作經驗年限）
y_pred = loaded_model.predict(X_test)                               # 使用加載的模型進行預測

# 視覺化結果
plt.scatter(X, y, label='data')                 # 繪製資料點
plt.plot(X, y_pred, 'r-', label='predicted')    # 繪製預測曲線
plt.legend()                                    # 顯示圖例
plt.show()                                      # 顯示圖表

# 測試模型
test_experience = np.array([5.5], dtype=np.float32)             # 測試數據（工作經驗年限）
predicted_salary = loaded_model.predict(test_experience)        # 使用加載的模型進行預測

print(f"Years of Experience: {test_experience}, Predicted Salary: {predicted_salary}")  # 輸出測試結果
