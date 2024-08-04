import pandas as pd                     # 匯入 pandas 庫，用於資料操作
import matplotlib.pyplot as plt         # 匯入 matplotlib 庫，用於繪圖
from ipywidgets import interact         # 匯入 ipywidgets 中的 interact，用於互動式小工具

# 從 CSV 文件讀取資料
data = pd.read_csv("./Salary_Data.csv")

# y = w*x + b
x = data["YearsExperience"]     # 將工作經驗年份的資料賦值給變數 x
y = data["Salary"]              # 將薪水的資料賦值給變數 y

# 繪製真實資料的散點圖
plt.scatter(x, y, marker="x", color="red")  # 使用紅色叉標記繪製散點圖
plt.title("Years of Experience vs Salary")  # 設置圖表標題
plt.xlabel("Years of Experience")           # 設置 X 軸標籤
plt.ylabel("Salary (k)")                    # 設置 Y 軸標籤
plt.show()                                  # 顯示圖表

# 定義函數，用於繪製線性預測函數
def plot_pred(w, b):
    # 根據線性函數計算預測的薪水
    y_pred = x * w + b
    # 繪製線性函數
    plt.plot(x, y_pred, color = "blue")             # 使用藍色繪製線性函數
    # 繪製真實資料的散點圖
    plt.scatter(x, y, marker = "x", color = "red")  # 使用紅色叉標記繪製散點圖
    plt.title("Years of Experience vs Salary")      # 設置圖表標題
    plt.xlabel("Years of Experience")               # 設置 X 軸標籤
    plt.ylabel("Salary (k)")                        # 設置 Y 軸標籤

    plt.xlim([0, 12])       # 設置 X 軸範圍
    plt.ylim([-20, 200])    # 設置 Y 軸範圍
    plt.legend()            # 顯示圖例
    plt.show()              # 顯示圖表

# 使用 w=0 和 b=10 繪製初始圖表
plot_pred(0, 10)

# 使用互動式小工具來改變 w 和 b 的值
interact(plot_pred, w = (-100, 100, 1), b = (-100, 100, 1))
