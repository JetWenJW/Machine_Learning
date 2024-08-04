import pandas as pd                                         # 匯入 pandas 庫，用於資料操作
import matplotlib.pyplot as plt                             # 匯入 matplotlib 庫，用於繪圖
from ipywidgets import interact                             # 匯入 ipywidgets 中的 interact，用於互動式小工具
import numpy as np                                          # 匯入 numpy 庫，用於數值計算
from sklearn.preprocessing import OneHotEncoder             # 匯入 OneHotEncoder，用於將類別變數轉換為獨熱編碼
from sklearn.preprocessing import StandardScaler            # 匯入 StandardScaler，用於特徵縮放
from sklearn.model_selection import train_test_split        # 匯入 train_test_split，用於資料集劃分

# Step 1: 準備資料
# 從 CSV 文件讀取資料
data = pd.read_csv("./Salary_Data2.csv")

# 從數據集中提取工作經驗年限 (x) 和薪水 (y)
x = data["YearsExperience"]
y = data["Salary"]

# 標籤編碼教育水平
data["EducationLevel"] = data["EducationLevel"].map({"高中以下": 0, "大學": 1, "碩士以上": 2})

# Step 2: 清理資料集
# 城市的獨熱編碼
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(data[["City"]])
city_encoded = onehot_encoder.transform(data[["City"]]).toarray()
data[["CityA", "CityB", "CityC"]] = city_encoded

# 刪除未使用的 City 和獨熱編碼的 CityC
data = data.drop(["City", "CityC"], axis=1)

# 組織特徵 x 和標籤 y
x = data[["YearsExperience", "EducationLevel", "CityA", "CityB"]]
y = data["Salary"]

# Step 3: 特徵工程
# 通過特徵縮放加速梯度下降
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Step 4: 資料分割
# 將資料分成訓練集和測試集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)

# Step 5: 建立模型
# 初始化權重和偏差
w = np.array([1, 2, 3, 4])
b = 1

# Step 6: 模型訓練
# 計算預測值
y_pred = (x_train * w).sum(axis=1) + b

# 定義成本函數
def compute_cost(x, y, w, b):
    y_pred = (x * w).sum(axis=1) + b    # 計算預測值
    cost = ((y - y_pred) ** 2).mean()   # 計算成本
    return cost

# 初始化權重和偏差
w = np.array([0, 2, 2, 4])
b = 0
compute_cost(x_train, y_train, w, b)    # 計算初始成本

# 定義梯度函數
def compute_gradient(x, y, w, b):
    y_pred = (x * w).sum(axis=1) + b            # 計算預測值
    w_gradient = np.zeros(x_train.shape[1])     # 初始化 w 的梯度
    b_gradient = (y_pred - y).mean()            # 計算 b 的梯度

    for i in range(x_train.shape[1]):
        w_gradient[i] = (x_train[:, i] * (y_pred - y)).mean()  # 計算 w 的梯度
    return w_gradient, b_gradient

# 設置 numpy 顯示選項
np.set_printoptions(formatter={'float': '{: .2e}'.format})

# 定義梯度下降優化函數
def gradient_descent(x, y, w_init, b_init, cost_function, gradient_function, run_iter, learning_rate, p_iter):
    # 存儲參數和成本歷史的列表
    c_hist = []
    w_hist = []
    b_hist = []
    w = w_init
    b = b_init
    
    # 梯度下降迭代
    for i in range(run_iter):
        # 計算梯度
        w_gradient, b_gradient = gradient_function(x, y, w, b)

        # 使用梯度和學習率更新參數
        w = w - learning_rate * w_gradient
        b = b - learning_rate * b_gradient
        
        # 使用更新後的參數計算成本
        cost = cost_function(x, y, w, b)
        
        # 存儲參數和成本歷史
        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)

        # 每 p_iter 次迭代打印進度
        if i % p_iter == 0:
            print(f"Iteration {i:5}: w = {w: .2e}, b = {b: .2e}, w_gradient = {w_gradient: .2e}, b_gradient = {b_gradient: .2e}, Cost = {cost: .2e}")
    
    # 返回最終參數和歷史
    return w, b, w_hist, b_hist, c_hist

# 初始化梯度下降參數
w_init = np.array([1, 2, 2, 4])
b_init = 1
run_iter = 10000
learning_rate = 0.001

# 執行梯度下降
w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x_train, y_train, w_init, b_init, compute_cost, compute_gradient, run_iter, learning_rate, p_iter=1000)

# Step 7: 模型評估
# 在測試集上驗證訓練好的模型
y_pred = (x_test * w_final).sum(axis=1) + b_final
print(pd.DataFrame({"y_pred": y_pred, "y_test": y_test}))

# 計算成本以評估預測質量
print(compute_cost(x_test, y_test, w_final, b_final))

# Step 8: 模型評估（此處暫不進行）

# Step 9: 模型部署（此處暫不進行）

# Step 10: 新資料預測
# 使用實際資料進行測試
x_real = np.array([[5.3, 2, 1, 0], [7.2, 0, 0, 1]])
x_real = scaler.transform(x_real)                   # 對新資料進行特徵縮放
y_real = (w_final * x_real).sum(axis=1) + b_final   # 預測新資料的結果
print(y_real)
