import numpy as np
import pandas as pd
import os

# 列出指定目錄中的所有文件
for dirname, _, filenames in os.walk('./mercari'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 讀取訓練和測試數據
train_df = pd.read_table('./mercari/train.tsv/train.tsv')
test_df = pd.read_table('./mercari/test.tsv/test.tsv')
print(train_df.shape, test_df.shape)  # 打印訓練和測試數據的形狀

# 查看訓練和測試數據的前幾行
print(train_df.head())
print(test_df.head())

# 去除價格低於3.0的數據
train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
print(train_df.shape)  # 打印去除後的數據形狀
print(train_df['price'].max())  # 打印價格的最大值
print(train_df['price'].min())  # 打印價格的最小值

# 繪製價格分佈直方圖
# 呈現偏態分布
import matplotlib.pyplot as plt
train_df['price'].hist()  # 繪製價格分佈的直方圖
train_df['price'].hist(range=(0, 100))  # 繪製價格範圍在(0, 100)內的直方圖

# 對價格取對數變換
# 透過對數轉換使其呈現常態分布
train_df["target"] = np.log1p(train_df.price)  # 對價格取對數變換
train_df["target"].hist()  # 繪製對數變換後的價格分佈直方圖

# 繪製價格分佈直方圖
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(train_df['price'], bins=50)  # 繪製價格分佈的直方圖
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

# 繪製價格範圍在 (0, 100) 之內的直方圖
plt.subplot(1, 2, 2)
plt.hist(train_df['price'], bins=50, range=(0, 100))  # 繪製價格範圍在(0, 100)內的直方圖
plt.title('Price Distribution (0 to 100)')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 對價格取對數變換
# 透過對數轉換使其呈現常態分布
train_df["target"] = np.log1p(train_df.price)  # 對價格取對數變換
plt.figure(figsize=(6, 4))
plt.hist(train_df["target"], bins=50)  # 繪製對數變換後的價格分佈直方圖
plt.title('Log-Transformed Price Distribution')
plt.xlabel('Log-Transformed Price')
plt.ylabel('Frequency')
plt.show()

# 分割類別名稱
# 原資料，子項目全部集中在同一欄並以'/'隔開
# 這裡將他們拆開，並個別存放到subcat_0，subcat_1，subcat_2中
def split_cat(text):
    try:
        return text.split('/')
    except:
        return ('No Label', 'No Label', 'No Label')

train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = \
    zip(*train_df['category_name'].apply(lambda x: split_cat(x)))

test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = \
    zip(*test_df['category_name'].apply(lambda x: split_cat(x)))

print(train_df.head())
print(test_df.head())

# 將品牌名稱中缺失值轉換成有意義的資料
# 合併訓練和測試數據
full_set = pd.concat([train_df, test_df])

# 獲取所有品牌名稱的集合
all_brands = set(full_set['brand_name'].values)

# 將缺失的品牌名稱填充為 'missing'
train_df['brand_name'] = train_df['brand_name'].fillna('missing')
test_df['brand_name'] = test_df['brand_name'].fillna('missing')

train_premissing = len(train_df.loc[train_df['brand_name'] == 'missing'])
test_premissing = len(test_df.loc[test_df['brand_name'] == 'missing'])

# 自定義函數來查找品牌名稱
def brandfinder(line):
    """
    Parameters:
        line (pd.Series): 包含品牌名稱和商品名稱的 Series
    Returns:
        str: 更新後的品牌名稱
    """
    brand = line.iloc[0]  # 使用 iloc[0] 根據位置取值，索取brand name
    name = line.iloc[1]   # 使用 iloc[1] 根據位置取值，索取商品名稱
    namesplit = name.split(' ') # 用空格，隔開商品名稱

    if brand == 'missing':
        for x in all_brands:
            return name
    if name in all_brands:
        return name
    return brand

# 更新品牌名稱
train_df['brand_name'] = train_df[['brand_name', 'name']].apply(brandfinder, axis=1)
test_df['brand_name'] = test_df[['brand_name', 'name']].apply(brandfinder, axis=1)

train_len = len(train_df.loc[train_df['brand_name'] == 'missing'])
test_len = len(test_df.loc[test_df['brand_name'] == 'missing'])
train_found = train_premissing - train_len
test_found = test_premissing - test_len

print(train_premissing)
print(train_found)
print(test_premissing)
print(test_found)

print(train_df.head())
full_df = pd.concat([train_df, test_df], sort=False)

# 填充缺失值
def fill_missing_values(df):
    df['category_name'] = df['category_name'].fillna('missing')
    df['brand_name'] = df['brand_name'].fillna('missing')
    df['item_description'] = df['item_description'].fillna('missing')
    df['item_description'] = df['item_description'].replace('No description yet', 'missing')
    return df

full_df = fill_missing_values(full_df)

# 標籤編碼(對文字資料進行編碼)
# Label Encoding
# 商品類別，商品名稱，3層級商品類別都只有一個單詞
# 所以可以進行 Enoding，由於有層級之分所以用Label Encoding
from sklearn.preprocessing import LabelEncoder
print("Processing category data...")

le = LabelEncoder()
le.fit(full_df['category_name'])
full_df['category'] = le.transform(full_df['category_name'])

le.fit(full_df['brand_name'])
full_df['brand_name'] = le.transform(full_df['brand_name'])

le.fit(full_df['subcat_0'])
full_df['subcat_0'] = le.transform(full_df['subcat_0'])

le.fit(full_df['subcat_1'])
full_df['subcat_1'] = le.transform(full_df['subcat_1'])

le.fit(full_df['subcat_2'])
full_df['subcat_2'] = le.transform(full_df['subcat_2'])

# 刪除標籤編碼器對象以釋放內存
del le
print(full_df.category.head())
print(full_df.brand_name.head())
print(full_df.subcat_0.head())
print(full_df.subcat_1.head())
print(full_df.subcat_2.head())

# 將文本數據轉換為序列
# 將商品名稱，商品敘述分解為單詞，進行編碼
from tensorflow.keras.preprocessing.text import Tokenizer

# 將文字轉分割整理成不重複的單詞
print("Transforming text data to sequence...")
raw_text = np.hstack([full_df['item_description'].str.lower(),
                      full_df['name'].str.lower(),
                      full_df['category_name'].str.lower()])
print('Sequences shape', raw_text.shape)

# 建立Tokenizer
print("Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

# 每個單詞對應一個索引，文字資料就轉換成數值向量(Label Encoding)
print("Transforming text to sequences...")
full_df['seq_item_description'] = tok_raw.texts_to_sequences(
    full_df['item_description'].str.lower())
full_df['seq_name'] = tok_raw.texts_to_sequences(
    full_df['name'].str.lower())
del tok_raw
print(full_df['seq_item_description'].head())
print(full_df['seq_name'].head())

# 填充序列
from tensorflow.keras.preprocessing.sequence import pad_sequences
print(pad_sequences(full_df['seq_item_description'], maxlen=80), '\n')
print(pad_sequences(full_df['seq_name'], maxlen=10))
