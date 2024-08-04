import pandas as pd                 # 匯入 pandas 庫，用於資料操作
import matplotlib.pyplot as plt     # 匯入 matplotlib 庫，用於繪圖
from ipywidgets import interact     # 匯入 ipywidgets 中的 interact，用於互動式小工具
import numpy as np                  # 匯入 numpy 庫，用於數值計算

# 從 CSV 文件讀取資料
data = pd.read_csv("./Salary_Data.csv")

# 從數據集中提取工作經驗年限 (x) 和薪水 (y)
x = data["YearsExperience"]
y = data["Salary"]

# 線性回歸參數的初始值
w = 10
b = 0

# 計算預測值 (y_pred) 和成本函數 (cost)
y_pred = w * x + b
cost = (y - y_pred) ** 2
cost.sum() / len(x)

# 定義函數，根據給定的參數 w 和 b 計算成本函數
def compute_cost(x, y, w, b):
    y_pred = w * x + b
    cost = (y - y_pred) ** 2
    cost = cost.sum() / len(x)
    return cost

# 用特定的 w 和 b 值計算成本的例子
compute_cost(x, y, 10, 10)

# 計算範圍從 -100 到 100 的 w 值的成本，b 固定為 0
costs = []
for w in range(-100, 101):
    cost = compute_cost(x, y, w, 0)
    costs.append(cost)

# 繪製範圍從 -100 到 100 的 w 值的成本函數，b 固定為 0
plt.plot(range(-100, 101), costs)
plt.title("Cost Function for b = 0, w = -100 to 100")
plt.xlabel("w")
plt.ylabel("Cost")
plt.show()

# 計算從 -100 到 100 的 w 和 b 值範圍的成本
ws = np.arange(-100, 101)
bs = np.arange(-100, 101)
costs = np.zeros((201, 201))

# 計算所有 w 和 b 組合的成本
i = 0
for w in ws:
    j = 0
    for b in bs:
        cost = compute_cost(x, y, w, b)
        costs[i, j] = cost
        j += 1
    i += 1

# 繪製 w 和 b 值範圍內成本的 3D 曲面圖
plt.figure(figsize=(10, 8))
ax = plt.axes(projection="3d")
ax.view_init(45, -120)              # 調整 3D 投影的視角
ax.xaxis.set_pane_color((0, 0, 0))  # 設置 X 軸面顏色
ax.yaxis.set_pane_color((0, 0, 0))  # 設置 Y 軸面顏色
ax.zaxis.set_pane_color((0, 0, 0))  # 設置 Z 軸面顏色

b_grid, w_grid = np.meshgrid(bs, ws)
ax.plot_surface(w_grid, b_grid, costs, alpha = 0.8)     # 繪製成本曲面圖

ax.set_title("Cost in w & b Variables")                 # 設置圖表標題
ax.set_xlabel("w")                                      # 設置 X 軸標籤
ax.set_ylabel("b")                                      # 設置 Y 軸標籤
ax.set_zlabel("Cost")                                   # 設置 Z 軸標籤

# 找到最小成本值並繪製它們
w_index, b_index = np.where(costs == np.min(costs))
ax.scatter(ws[w_index], bs[b_index], costs[w_index, b_index], color = "darkred", s = 40)  # 繪製最小成本點

plt.show()  # 顯示圖表

# # 結論:
# 1.準備好一堆資料
# 2.找出最佳回歸直線
# 3.利用最有效率的方法找到最佳回歸直線(Gradient Descent)

# # Machine Learning:
# 1.準備資料
# 2.設定模型
# 3.設定cost function
# 4.設定Optimizer
