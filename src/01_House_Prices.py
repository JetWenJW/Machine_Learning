# 載入pandas套件，用於資料處理
import pandas as pd

# 讀取訓練資料
train = pd.read_csv("./house_price/train.csv")
# 讀取測試資料
test = pd.read_csv("./house_price/test.csv")

# 打印出訓練資料的形狀
print('train shape: ', train.shape)
# 打印出測試資料的形狀
print('test shape: ', test.shape)

# 顯示訓練資料的基本資訊
train.info()

# 載入其他所需的套件
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# 將SalePrice進行對數轉換，使資料更接近常態分布
prices = pd.DataFrame(
    {'price': train['SalePrice'],
     'log(price + 1)': np.log1p(train['SalePrice'])
    })
# 打印轉換後的資料
print(prices)

# 打印對數轉換前後的偏度
print('price skew : ', skew(prices['price']))
# 打印對數轉換後的偏度
print('log(price + 1) skew : ', skew(prices['log(price + 1)']))

# 將對數轉換前後的數值以直方圖顯示
plt.rcParams['figure.figsize'] = (12.0, 6.0)
# 顯示直方圖
prices.hist()
plt.show()

# 將SalePrice進行對數轉換(剛剛只是暫存於Dataframe中，目的是打印出來觀察)
train["SalePrice"] = np.log1p(train["SalePrice"])

# 合併訓練資料和測試資料(選擇從MSSubClass ~ SaleCondition的列)
all_data = pd.concat((
    train.loc[:, 'MSSubClass':'SaleCondition'],
    test.loc[:, 'MSSubClass':'SaleCondition']
))

# 打印合併後的資料形狀
print(all_data.shape)
# 打印合併後的資料
print(all_data)

# 獲取數值型特徵欄位(選取數據類行為 非Object的 節取出來再用index獲取篩選後列的標籤，即列名)
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# 打印數值型特徵欄位
print('-----Column of non-object type-----')
print(numeric_feats)

# 計算數值型特徵欄位的偏度
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
# 打印數值型特徵欄位的偏度
print('-----Skewness of non-object type Column-----')
print(skewed_feats)

# 選取偏度大於0.75的特徵
skewed_feats = skewed_feats[skewed_feats > 0.75]
# 打印偏度大於0.75的特徵
print('-----Skewness Greater than 0.75-----')
print(skewed_feats)

# 將偏度大於0.75的特徵進行對數轉換
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
# 打印轉換後的特徵
print(all_data[skewed_feats])

# One-Hot Encoding 將分類特徵轉換為數值特徵
cc_data = pd.get_dummies(train['LotShape'])
# 添加LotShape特徵到資料集中
cc_data['LotShape'] = train['LotShape']
# 打印前20筆資料
print(cc_data[:20])

# 將所有資料進行One-Hot Encoding並填補缺失值
all_data = pd.get_dummies(all_data)
# 填補缺失值
all_data = all_data.fillna(all_data[:train.shape[0]].mean())

# 分割訓練資料和測試資料
x_train = all_data[:train.shape[0]]
# 分割測試資料
x_test = all_data[train.shape[0]:]
# 訓練資料的目標值
y = train.SalePrice

# 定義交叉驗證函數
from sklearn.model_selection import cross_val_score
def rmse_cv(model):
    # 計算均方誤差(cv = 5，代表Cross Validation分為5個Folds)
    rmse = np.sqrt(-cross_val_score(model, x_train, y, scoring = "neg_mean_squared_error", cv = 5))
    return rmse

# 建立Ridge模型並進行交叉驗證
from sklearn.linear_model import Ridge
# 初始化Ridge模型
model = Ridge()
# 設定不同的正則化強度
alphas = [0.05, 0.1, 0.5, 1, 5, 10, 15, 30, 50, 75]

# 計算不同正則化強度下的均方誤差
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]

# 將結果轉為Series(以alphas建立對應索引)
cv_ridge = pd.Series(cv_ridge, index=alphas)

# 打印Ridge模型的RMSE損失
print('Ridge RMSE loss :')
print(cv_ridge, '\n')

# 打印Ridge模型的平均RMSE損失
print('Ridge RMSE loss Mean :')
print(cv_ridge.mean())

# 顯示不同正則化強度下的RMSE
plt.figure(figsize=(10, 5))
# 畫出RMSE曲線
plt.plot(cv_ridge)
plt.grid()
plt.title('Validation - by regularization strength')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.show()

# 建立Lasso模型並進行交叉驗證
from sklearn.linear_model import LassoCV
# 初始化Lasso模型
model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(x_train, y)

# 打印Lasso模型的RMSE損失
print('Lasso regression RMSE loss:')
print(rmse_cv(model_lasso))

# 打印Lasso模型(不同aplpha值)的平均RMSE損失
print('Average Loss : ', rmse_cv(model_lasso).mean())
# 打印Lasso模型(不同aplpha值)的最小RMSE損失
print('Minimum Loss : ', rmse_cv(model_lasso).min())
# 打印Lasso模型選擇的最佳alpha值
print('Best alpha : ', model_lasso.alpha_)

# 使用XGBoost進行建模和交叉驗證
import xgboost as xgb
# 創建DMatrix
dtrain = xgb.DMatrix(x_train, label=y)
# 設定XGBoost參數
params = {"max_depth": 3, "eta": 0.1}

# 進行交叉驗證(Cross Validation)
cross_val = xgb.cv(
    params,                    # 模型超參數
    dtrain,                    # 訓練數據（DMatrix 對象）
    num_boost_round=1000,      # 最大迭代輪數（即決策樹的數量）
    early_stopping_rounds=50   # 早停輪數：如果在 50 輪中模型性能不再提高，則提前停止訓練
)

# 顯示XGBoost交叉驗證結果
plt.figure(figsize=(8, 6))
# 畫出訓練集的RMSE
plt.plot(cross_val.loc[30:, ["train-rmse-mean"]], linestyle='--', label='Train')
# 畫出測試集的RMSE
plt.plot(cross_val.loc[30:, ["test-rmse-mean"]], label='Validation')
plt.grid()
plt.xlabel('num_boost_round')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# 訓練XGBoost模型
import xgboost as xgb

# 初始化 XGBRegressor 模型
model_xgb = xgb.XGBRegressor(
    n_estimators=410,          # 設置樹的數量，即模型中要訓練的決策樹的數量
    max_depth=3,               # 設置每棵決策樹的最大深度，用來控制模型的複雜度
    learning_rate=0.1          # 設置學習率，控制每棵樹的貢獻度
)

# 訓練模型
model_xgb.fit(x_train, y)
# 打印XGBoost模型的RMSE損失
print('xgboost RMSE loss: ')

# 將(Cross Validation = 5)結果取平均
print(rmse_cv(model_xgb).mean())

# 結合Lasso和XGBoost模型的預測結果(因為一開始有做對數轉換，這裡用指數還回原數值)
lasso_preds = np.expm1(model_lasso.predict(x_test))

# XGBoost的預測結果(因為一開始有做對數轉換，這裡用指數還回原數值)
xgb_preds = np.expm1(model_xgb.predict(x_test))

# 結合兩個模型的預測結果(Ensemble)
preds = lasso_preds * 0.7 + xgb_preds * 0.3

# 將預測結果保存到CSV檔案
solution = pd.DataFrame({"id": test.Id, "SalePrice": preds})
solution.to_csv("ensemble_sol.csv", index=False)
