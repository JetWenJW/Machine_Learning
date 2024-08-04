from datetime import datetime

# 記錄程式開始時間
start_real = datetime.now()

import pandas as pd

# 讀取訓練和測試資料
train_df = pd.read_table('./mercari/train.tsv/train.tsv')
test_df = pd.read_table('./mercari/test.tsv/test.tsv')

# 印出訓練和測試資料的形狀
print(train_df.shape, test_df.shape)

# 去除價格低於3的訓練數據
train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
print(train_df.shape)

def wordCount(text):
    """
    計算名稱或描述中的單詞數量
    Parameters:
        text(str): 商品名稱、商品敘述 
    """
    try:
        if text == 'No description yet':
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except:
        return 0

# 計算名稱和描述的字數並加入數據框
# 把 name 單辭數量加入 name_len
train_df['name_len'] = train_df['name'].apply(lambda x: wordCount(x))
test_df['name_len'] = test_df['name'].apply(lambda x: wordCount(x))

# 把 item_description 單辭數量加入 desc_len
train_df['desc_len'] = train_df['item_description'].apply(lambda x: wordCount(x))
test_df['desc_len'] = test_df['item_description'].apply(lambda x: wordCount(x))

import numpy as np

# 將價格取對數作為目標變量(使其越接近常態分布)
train_df["target"] = np.log1p(train_df.price)

def split_cat(text):
    """
    將類別名稱分割成多個子類別
    Parameters:
        text(str): 類別名稱
        · 使用 / 分割類別名稱
        · 若資料不存在 / 時則傳回"No Label" 
    """
    try: 
        return text.split("/")
    except:
        return("No Label", "No Label", "No Label")

# 將類別名稱分割成多個子類別並加入數據框
train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = zip( * train_df['category_name'].apply(lambda x: split_cat(x)))
test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = zip( * test_df['category_name'].apply(lambda x: split_cat(x)))



# 合併訓練和測試資料以提取所有品牌名稱
full_set = pd.concat([train_df, test_df])
all_brands = set(full_set['brand_name'].values)

# 填充缺失的品牌名稱(把 brand_name 中 NaN ，以 missing 做填充)
train_df['brand_name'] = train_df['brand_name'].fillna(value='missing')
test_df['brand_name'] = test_df['brand_name'].fillna(value='missing')

# 計算缺失品牌名稱的數量
train_premissing = len(train_df.loc[train_df['brand_name'] == 'missing'])
test_premissing = len(test_df.loc[test_df['brand_name'] == 'missing'])

def brandfinder(line):
    """
    將缺失的品牌名稱替換為商品名稱中的品牌
    Parameters: 
        line(str): 品牌名稱
        · 將品牌名稱的'missing'替換為商品名稱：
            當'missing'的商品名稱單詞存在於品牌清單中時
        · 將品牌名稱替換為商品名稱:
            當商品名稱與品牌清單中的名稱完全一致時
        · 維持現有品牌名稱:
            商品名稱與品牌清單的名稱不一致品牌名稱雖為'missing'，但商品名稱的單詞不在品牌清單內
    """
    brand = line[0]
    name = line[1]
    namesplit = name.split(' ')

    if brand == 'missing':
        for x in namesplit:
            if x in all_brands:
                return name
    if name in all_brands:
        return name
    return brand

# 計算修復後的品牌名稱數量
train_found = train_premissing - len(train_df.loc[train_df['brand_name'] == 'missing'])
test_found = test_premissing - len(test_df.loc[test_df['brand_name'] == 'missing'])

# 印出改寫後的缺失值數量
print(train_premissing)
print(train_found)
print(test_premissing)
print(test_found)

from sklearn.model_selection import train_test_split
import gc

# 切分訓練資料為訓練集和驗證集
train_dfs, dev_dfs = train_test_split(train_df, random_state = 123, train_size = 0.99, test_size = 0.01)

# 打印訓練集、驗證集和測試集的數量
n_trains = train_dfs.shape[0]
n_devs = dev_dfs.shape[0]
n_tests = test_df.shape[0]
print('Training :', n_trains, 'examples')
print('Validating :', n_devs, 'examples')
print('Testing :', n_tests, 'examples')

# 釋放記憶體
del train_df
gc.collect()

# 合併所有資料集
full_df = pd.concat([train_dfs, dev_dfs, test_df])

# 填充缺失值
def fill_missing_values(df):
    df['category_name'] = df['category_name'].fillna('missing')
    df['brand_name'] = df['brand_name'].fillna('missing')
    df['item_description'] = df['item_description'].fillna('missing')
    df['item_description'] = df['item_description'].replace('No description yet', 'missing')
    return df

full_df = fill_missing_values(full_df)



# 對商品類別，品牌名稱，3層級商品類別的文字進行 Label Encoding
from sklearn.preprocessing import LabelEncoder
print("Processing category data...")

# 將類別、品牌和子類別轉換為數值
le = LabelEncoder()
le.fit(full_df.category_name)
full_df['category'] = le.transform(full_df.category_name)

le.fit(full_df.brand_name)
full_df.brand_name = le.transform(full_df.brand_name)

le.fit(full_df.subcat_0)
full_df.subcat_0 = le.transform(full_df.subcat_0)

le.fit(full_df.subcat_1)
full_df.subcat_1 = le.transform(full_df.subcat_1)

le.fit(full_df.subcat_2)
full_df.subcat_2 = le.transform(full_df.subcat_2)

# 釋放記憶體
del le
gc.collect()

from tensorflow.keras.preprocessing.text import Tokenizer

print("Transforming text data to sequences...")

# 將文本數據轉換為序列
# 把商品名稱，商品敘述分解為單詞
raw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower(), full_df.category_name.str.lower()])
print('sequence shape', raw_text.shape)

# 創建並擬合分詞器(Tokenizer)
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

print("Transforming text to sequences...")

# 將文本轉換為序列，再進行 Label Encoding
full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())
full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())

# 釋放記憶體
del tok_raw
gc.collect()

# 定義各種特徵的最大長度
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_CATEGORY_SEQ = 8

# 定義最大文本、類別、品牌等數值
MAX_TEXT = np.max([np.max(full_df.seq_name.max()), np.max(full_df.seq_item_description.max())]) + 100
MAX_CATEGORY = np.max(full_df.category.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1

MAX_SUBCAT_0 = (full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = (full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = (full_df.subcat_2.max()) + 1

from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_run_data(dataset):
    # 準備訓練、驗證和測試數據
    x = {
        'name': pad_sequences(dataset.seq_name, maxlen = MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen = MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category': np.array(dataset.brand_name),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
        'subcat_0': np.array(dataset.subcat_0),
        'subcat_1': np.array(dataset.subcat_1),
        'subcat_2': np.array(dataset.subcat_2),
    }
    return x

# 準備訓練、驗證和測試數據
train = full_df[:n_trains]
dev = full_df[n_trains: n_trains + n_devs]
test = full_df[n_trains + n_devs:]

X_train = get_run_data(train)
Y_train = train.target.values.reshape(-1, 1)

X_dev = get_run_data(dev)
Y_dev = dev.target.values.reshape(-1, 1)

X_test = get_run_data(test)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, Flatten
from tensorflow.keras.layers import concatenate, GRU
from tensorflow.keras.optimizers import Adam

np.random.seed(123)

def rmsle(Y, Y_pred):
    # 定義RMSLE評估指標
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y)))

def new_rnn_model(lr = 0.001, decay = 0.0):
    """
    生成循環神經網路模型
    Parameters:
        lr: 學習率
        decay: 學習率的衰減
    """
    # 定義模型的輸入
    name = Input(shape = [X_train["name"].shape[1]], name = "name")
    item_desc = Input(shape = [X_train["item_desc"].shape[1]], name = "item_desc")
    brand_name = Input(shape = [1], name = "brand_name")
    item_condition = Input(shape = [1], name = "item_condition")
    num_vars = Input(shape = [X_train["num_vars"].shape[1]], name = "num_vars")
    name_len = Input(shape = [1], name = "name_len")
    desc_len = Input(shape = [1], name = "desc_len")
    subcat_0 = Input(shape = [1], name = "subcat_0")
    subcat_1 = Input(shape = [1], name = "subcat_1")
    subcat_2 = Input(shape = [1], name = "subcat_2")

    # 定義嵌入層(Word Embedding)
    # 將單詞或詞彙轉換為數值向量。
    # 這些向量可以捕捉到詞彙之間的語義和語法關係，
    # 使得機器學習模型能夠更好地處理和理解文本數據。
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
    emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
    emb_subcat_0 = Embedding(MAX_SUBCAT_0, 10)(subcat_0)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 10)(subcat_2)

    # 定義GRU層
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)

    # 定義主模型
    main_1 = concatenate([
        Flatten()(emb_brand_name),
        Flatten()(emb_item_condition),
        Flatten()(emb_desc_len),
        Flatten()(emb_name_len),
        Flatten()(emb_subcat_0),
        Flatten()(emb_subcat_1),
        Flatten()(emb_subcat_2),
        rnn_layer1,
        rnn_layer2,
        num_vars
    ])
    
    # 定義密集層和Dropout層
    main_1 = Dropout(0.1)(Dense(512, kernel_initializer = 'normal', activation = 'relu')(main_1))
    main_1 = Dropout(0.1)(Dense(256, kernel_initializer = 'normal', activation = 'relu')(main_1))
    main_1 = Dropout(0.1)(Dense(128, kernel_initializer = 'normal', activation = 'relu')(main_1))
    main_1 = Dropout(0.1)(Dense(64, kernel_initializer = 'normal', activation = 'relu')(main_1))
    
    # 定義輸出層
    output = Dense(1, activation = "linear")(main_1)

    # 創建模型
    model = Model(inputs = [
        name, item_desc, brand_name, item_condition, num_vars, desc_len, name_len, subcat_0, subcat_1, subcat_2
    ], outputs = output)
    
    # 編譯模型
    model.compile(loss = 'mse', optimizer = Adam(learning_rate = lr, decay = decay))
    
    return model

# 生成模型並打印模型摘要
model = new_rnn_model()    
model.summary()

# 釋放記憶體
del model
gc.collect()

# 設定批次大小和訓練週期
BATCH_SIZE = 512 * 2
epochs = 3

# 計算學習率衰減
exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init = 0.005
lr_fin = 0.001
lr_decay = exp_decay(lr_init, lr_fin, steps)

# 生成並訓練模型
rnn_model = new_rnn_model(lr = lr_init, decay = lr_decay)
rnn_model.fit(X_train, Y_train, epochs = epochs, batch_size = BATCH_SIZE, validation_data = (X_dev, Y_dev), verbose = 1)

# 評估模型在驗證數據上的表現
print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size = BATCH_SIZE)
print("RMSLE error: ", rmsle(Y_dev, Y_dev_preds_rnn))

# 預測測試數據
rnn_preds = rnn_model.predict(X_test, batch_size = BATCH_SIZE, verbose = 1)
rnn_preds = np.expm1(rnn_preds)

# 釋放記憶體
del rnn_model
gc.collect()

# 記錄程式結束時間並計算執行時間
stop_real = datetime.now()
execution_time_real = stop_real - start_real
print(execution_time_real)
