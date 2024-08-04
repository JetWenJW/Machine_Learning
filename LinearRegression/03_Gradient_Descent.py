import pandas as pd                 # 匯入 pandas 庫，用於資料操作
import matplotlib.pyplot as plt     # 匯入 matplotlib 庫，用於繪圖
from ipywidgets import interact     # 匯入 ipywidgets 中的 interact，用於互動式小工具
import numpy as np                  # 匯入 numpy 庫，用於數值計算

# 從 CSV 文件讀取資料
data = pd.read_csv("./Salary_Data.csv")

# 從數據集中提取工作經驗年限 (x) 和薪水 (y)
x = data["YearsExperience"]
y = data["Salary"]

# 定義函數，用於計算給定參數 w 和 b 的梯度
def compute_gradient(x, y, w, b):
    w_gradient = (x * (w * x + b - y)).mean()   # 計算 w 的梯度
    b_gradient = ((w * x + b - y)).mean()       # 計算 b 的梯度
    return w_gradient, b_gradient

# 定義函數，用於計算給定參數 w 和 b 的成本函數
def compute_cost(x, y, w, b):
    y_pred = w * x + b          # 計算預測值
    cost = (y - y_pred) ** 2    # 計算誤差平方和
    cost = cost.sum() / len(x)  # 計算平均誤差平方和
    return cost

# 定義梯度下降優化的函數
def gradient_descent(x, y, w_init, b_init, cost_function, gradient_function, run_iter, learning_rate, p_iter):
    # 用於儲存參數和成本歷史的列表
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
        
        # 儲存參數和成本歷史
        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)

        # 每 p_iter 次迭代打印一次進度
        if i % p_iter == 0:
            print(f"Iteration {i:5}: w = {w: .2e}, b = {b: .2e}, w_gradient = {w_gradient: .2e}, b_gradient = {b_gradient: .2e}, Cost = {cost: .2e}")
    
    # 返回最終參數和歷史
    return w, b, w_hist, b_hist, c_hist

# 梯度下降的初始參數
w_init = 0
b_init = 0
learning_rate = 1.0e-3
run_iter = 20000
p_iter = 1000

# 執行梯度下降找到最佳 (w, b)
w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x, y, w_init, b_init, compute_cost, compute_gradient, run_iter, learning_rate, p_iter)

# 打印最終參數 (w, b)
print(f"Final (w, b) = ({w_final:.2f}, {b_final:.2f})")

# 繪製迭代次數與成本的關係圖
plt.plot(np.arange(0, 100), c_hist[:100])
plt.title("Iteration vs Cost")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# 預測工作經驗為 3.5 年的薪水
print(f"Year Exp: 3.5 -> Predicted Salary: {w_final*3.5 + b_final:.1f}K")

# w 和 b 的值範圍
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
ax.plot_surface(w_grid, b_grid, costs, alpha=0.3)   # 繪製成本曲面圖

ax.set_title("Cost in w & b Variables")             # 設置圖表標題
ax.set_xlabel("w")                                  # 設置 X 軸標籤
ax.set_ylabel("b")                                  # 設置 Y 軸標籤
ax.set_zlabel("Cost")                               # 設置 Z 軸標籤

# 找到最小成本值並繪製它們
w_index, b_index = np.where(costs == np.min(costs))
ax.scatter(ws[w_index], bs[b_index], costs[w_index, b_index], color="darkred", s=40)  # 繪製最小成本點

# 在 3D 投影上繪製初始值
ax.scatter(w_hist[0], b_hist[0], c_hist[0], color="green", s=40)

# 在 3D 投影上繪製最小線
ax.plot(w_hist, b_hist, c_hist)

plt.show()

# # 結論:
# 1.準備好一堆資料
# 2.找出最佳回歸直線
# 3.利用最有效率的方法找到最佳回歸直線(Gradient Descent)

# # 機器學習:
# 1.準備資料
# 2.設定模型
# 3.設定成本函數
# 4.設定優化器
