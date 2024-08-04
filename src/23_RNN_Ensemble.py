from datetime import datetime
start_real = datetime.now()

import pandas as pd
train_df = pd.read_table('./mercari/train.tsv/train.tsv')
test_df = pd.read_table('./mercari/test.tsv/test.tsv')
print(train_df.shape, test_df.shape)

train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
train_df.shape

def wordCount(text):
    try:
        if text == 'No description yet':
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except:
        return 0
train_df['name_len'] = train_df['name'].apply(lambda x : wordCount(x))
test_df['name_len'] = test_df['name'].apply(lambda x : wordCount(x))

train_df['desc_len'] = train_df['item_description'].apply(lambda x : wordCount(x))
test_df['desc_len'] = test_df['item_description'].apply(lambda x : wordCount(x))


import numpy as np

train_df["target"] = np.log1p(train_df.price)

def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")

train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = \
    zip( * train_df['category_name'].apply(lambda x: split_cat(x)))

test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = \
    zip( * test_df['category_name'].apply(lambda x: split_cat(x)))

full_set = pd.concat([train_df, test_df])
all_brands = set(full_set['brand_name'].values)

train_df['brand_name'] = train_df['brand_name'].fillna(value = 'missing')
test_df['brand_name'] = test_df['brand_name'].fillna(value = 'missing')

train_premissing = len(train_df.loc[train_df['brand_name'] == 'missing'])
test_premissing = len(test_df.loc[test_df['brand_name'] == 'missing'])

def brandfinder(line):
    
    """
    Parameters: line(str): 品牌名稱
    · 將品牌名稱的'missing'替換為商品名稱：
        當'missing'的商品名稱單詞存在於品牌清單中時
    · 將品牌名稱替換為商品名稱:
        當商品名稱與品牌清單中的名稱完全一致時
    · 維持現有品牌名稱:
        商品名稱與品牌清單的名稱不一致品牌名稱雖為'missing'，但商品名稱的單詞不在品牌清單內
    """
    brand = line[0]                 # 第 1 欄為品牌名稱
    name = line[1]                  # 第 2 欄為商品名稱
    namesplit = name.split(' ')     # 使用空格分割商品名稱

    if brand == 'missing':          # 是缺失值
        for x in namesplit:         # 取出從商品名稱分割出來的單詞
            if x in all_brands:
                return name         # 商品名稱單詞存在於品牌清單中，則傳回商品名稱單詞
    if name in all_brands:          # 不是缺失值
        return name                 # 商品名稱若存在於品牌清單中，則傳回商品名稱

    return brand                    # 都沒有一致的話就傳回品牌名稱

train_df['brand_name'] = train_df[['brand_name', 'name']].apply(brandfinder, axis = 1)
test_df['brand_name'] = test_df[['brand_name', 'name']].apply(brandfinder, axis = 1)

train_found = train_premissing - len(train_df.loc[train_df['brand_name'] == 'missing'])
test_found = test_premissing - len(test_df.loc[test_df['brand_name'] == 'missing'])

print(train_premissing)
print(train_found)
print(test_premissing)
print(test_found)


from sklearn.model_selection import train_test_split
import gc

train_dfs, dev_dfs = train_test_split(train_df,
                                      random_state = 123,
                                      train_size = 0.99,
                                      test_size = 0.01)

n_trains = train_dfs.shape[0]
n_devs = dev_dfs.shape[0]
n_tests = test_df.shape[0]

print('Training : ', n_trains, 'example')
print('Validating : ', n_devs, 'example')
print('Testing : ', n_tests, 'example')

del train_df
gc.collect()

full_df = pd.concat([train_dfs, dev_dfs, test_df])

# 填充缺失值
def fill_missing_values(df):
    df['category_name'] = df['category_name'].fillna('missing')
    df['brand_name'] = df['brand_name'].fillna('missing')
    df['item_description'] = df['item_description'].fillna('missing')
    df['item_description'] = df['item_description'].replace('No description yet', 'missing')
    return df

full_df = fill_missing_values(full_df)

from sklearn.preprocessing import LabelEncoder

print("Processing category data...")

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
del le
gc.collect()

from tensorflow.keras.preprocessing.text import Tokenizer
print("Transforming text data to Sequences...")
raw_text = np.hstack([full_df.item_description.str.lower(),
                      full_df.name.str.lower(),
                      full_df.category_name.str.lower()])
print('sequences shape', raw_text.shape)
print("Fitting tokenizer...")

tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

print("Transforming text to Sequences...")
full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())
full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())

del tok_raw
gc.collect()

MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_CATEGORY_SEQ = 8
MAX_TEXT = np.max([np.max(full_df.seq_name.max()),
                   np.max(full_df.seq_item_description.max())]) + 100
MAX_CATEGORY = np.max(full_df.category.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1

MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1

from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_rnn_data(dataset):
    """ 
    將輸入的資料放入dict物件後傳回  
    Parameter: 
        dataset: 全部資料 
    """ 
    X = {
        'name': pad_sequences(dataset.seq_name,
                               maxlen = MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description,
                                   maxlen = MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category': np.array(dataset.category),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
        'subcat_0': np.array(dataset.subcat_0),
        'subcat_1': np.array(dataset.subcat_1),
        'subcat_2': np.array(dataset.subcat_2),
        }    
    return X

train = full_df[:n_trains]
dev = full_df[n_trains: n_trains + n_devs]
test = full_df[n_trains + n_devs:]

X_train = get_rnn_data(train)
Y_train = train.target.values.reshape(-1, 1)

X_dev = get_rnn_data(dev)
Y_dev = dev.target.values.reshape(-1, 1)

X_test = get_rnn_data(test)

del full_df
gc.collect()


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, Flatten
from tensorflow.keras.layers import concatenate, GRU
from tensorflow.keras.optimizers import Adam

np.random.seed(123)

def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.squrt(np.mean(np.square(Y_pred - Y)))

def new_rnn_model(learning_rate = 0.001, decay = 0.0):
    """
    生成循環型類神經網路模型
    Parameters:
        lr: 學習率
        decay: 學習率的衰減
    """
    name            = Input(shape = [X_train["name"].shape[1]], name = "name")
    item_desc       = Input(shape = [X_train["item_desc"].shape[1]], name = "item_desc")
    brand_name      = Input(shape = [1], name = "brand_name")
    item_condition  = Input(shape = [1], name = "item_condition")
    num_vars        = Input(shape = [X_train["num_vars"].shape[1]], name = "num_vars")

    name_len        = Input(shape = [1], name = "name_len")
    desc_len        = Input(shape = [1], name = "desc_len")

    subcat_0        = Input(shape = [1], name = "subcat_0")
    subcat_1        = Input(shape = [1], name = "subcat_1")
    subcat_2        = Input(shape = [1], name = "subcat_2")

    emb_name            = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc       = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name      = Embedding(MAX_BRAND, 10)(brand_name)
    emb_item_condition  = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_desc_len        = Embedding(MAX_DESC_LEN, 5)(desc_len)
    emb_name_len        = Embedding(MAX_NAME_LEN, 5)(name_len)

    emb_subcat_0        = Embedding(MAX_SUBCAT_0, 10)(subcat_0)
    emb_subcat_1        = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2        = Embedding(MAX_SUBCAT_2, 10)(subcat_2)

    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)
    

    main_1 = concatenate([Flatten()(emb_brand_name),
                          Flatten()(emb_item_condition),
                          Flatten()(emb_desc_len),
                          Flatten()(emb_name_len),
                          Flatten()(emb_subcat_0),
                          Flatten()(emb_subcat_1),
                          Flatten()(emb_subcat_2),
                          rnn_layer1,
                          rnn_layer2,
                          num_vars])
    
    main_1 = Dropout(0.1)(Dense(512,
                                kernel_initializer = 'normal',
                                activation = 'relu')(main_1))
    main_1 = Dropout(0.1)(Dense(256,
                                kernel_initializer = 'normal',
                                activation = 'relu')(main_1))
    main_1 = Dropout(0.1)(Dense(128,
                                kernel_initializer = 'normal',
                                activation = 'relu')(main_1))
    main_1 = Dropout(0.1)(Dense(64,
                                kernel_initializer = 'normal',
                                activation = 'relu')(main_1))
    

    output = Dense(1, activation = "linear")(main_1)
    model = Model(inputs = [name,
                            item_desc,
                            brand_name,
                            item_condition,
                            num_vars,
                            desc_len,
                            name_len,
                            subcat_0,
                            subcat_1,
                            subcat_2],
                    outputs = output)
    
    model.compile(loss = 'mse',
                  optimizer = Adam(learning_rate = learning_rate, decay = decay))
    return model

model = new_rnn_model()
model.summary()

del model
gc.collect()

BATCH_SIZE = 512 * 2
epochs = 3
exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init = 0.005
lr_fin = 0.001
lr_decay = exp_decay(lr_init, lr_fin, steps)

rnn_model = new_rnn_model(learning_rate = lr_init, decay = lr_decay)
rnn_model.fit(X_train,
              Y_train,
              epochs = epochs,
              batch_size = BATCH_SIZE,
              validation_data = (X_dev, Y_dev),
              verbose = 1)

print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev,
                                    batch_size = BATCH_SIZE)
print("RMSLE Error: ", rmsle(Y_dev, Y_dev_preds_rnn))

rnn_preds = rnn_model.predict(X_test,
                              batch_size = BATCH_SIZE,
                              varbose = 1)
rnn_preds = np.expm1(rnn_preds)
del rnn_model
gc.collect()

full_df2 = pd.concat([train_dfs, dev_dfs, test_df])

print("Handling missing values...")

full_df2['category_name'] = full_df2['category_name'].fillna('missing').astype(str)
full_df2['subcat_0'] = full_df2['subcat_0'].astype(str)
full_df2['subcat_1'] = full_df2['subcat_1'].astype(str)
full_df2['subcat_2'] = full_df2['subcat_2'].astype(str)

full_df2['brand_name'] = full_df2['brand_name'].fillna('missing').astype(str)
full_df2['shipping'] = full_df2['shipping'].astype(str)

full_df2['item_condition_id'] = full_df2['item_condition_id'].astype(str)

full_df2['desc_len'] = full_df2['desc_len'].astype(str)
full_df2['name_len'] = full_df2['name_len'].astype(str)
full_df2['item_description'] = full_df2['item_description'].fillna('No description yet').astype(str)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

print("Vectorizing data...")
default_preprocessor = CountVectorizer().build_preprocessor()

def build_preprocessor(field):
    """ 
    取的指定欄位的索引
    傳回製作Token count矩陣的CountVectorizer
    Parameter:全連接資料框架的欄位名稱
    """
    field_idx = list(full_df2.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(ngram_range = (1, 2),
                             max_features = 5000,
                             preprocessor = build_preprocessor('name'))),
    ('subcat_0', CountVectorizer(token_pattern = '.+',
                                 prerocessor= build_preprocessor('subcat_0'))),
    ('subcat_1', CountVectorizer(token_pattern = '.+',
                                 prerocessor= build_preprocessor('subcat_1'))),
    ('subcat_2', CountVectorizer(token_pattern = '.+',
                                 prerocessor= build_preprocessor('subcat_2'))),
    ('brand_name', CountVectorizer(token_pattern = '.+',
                                 prerocessor= build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(token_pattern = '\d+',
                                 prerocessor= build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(token_pattern = '\d+',
                                 prerocessor= build_preprocessor('item_condition_id'))),
    ('desc_len', CountVectorizer(token_pattern = '\d+',
                                 prerocessor= build_preprocessor('desc_len'))),
    ('name_len', CountVectorizer(token_pattern = '\d+',
                                 prerocessor= build_preprocessor('name_len'))),
    ('item_description', TfidfVectorizer(ngram_range = (1, 3),
                                         max_features = 5000,
                                         preprocessor = build_preprocessor('item_description'))),])

X = vectorizer.fit_transform(full_df2.values)
del vectorizer
gc.collect()


X_train = X[:n_trains]
T_train = train_dfs.target.values.reshape(-1, 1)

X_dev = X[n_trains: n_trains + n_devs]
Y_dev = dev_dfs.target.values.reshape(-1, 1)

X_test = X[n_trains + n_devs:]

print('X:', X.shape)
print('X_train: ', X_train.shape)
print('X_dev: ', X_dev.shape)
print('X_test: ', X_test.shape)
print('Y_train: ', Y_train.shape)
print('Y_dev', Y_dev.shape)


from sklearn.linear_model import Ridge, RidgeCV

print("Fitting Ridge model on training examples...")
ridge_model = Ridge(solver = 'auto',
                     fit_intercept = True,
                     alpha = 1.0,
                     max_iter = 200,
                     normalize = False,
                     tol = 0.01,
                     random_state = 1)

ridge_modelCV = RidgeCV(fit_intercept = True,
                        alphas = [5.0],
                        normalize = False,
                        cv = 2,
                        scoring = 'neg_mean_squred_error')

ridge_model.fit(X_train, Y_train)
ridge_modelCV.fit(X_train, Y_train)

Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
print('Ridge model RMSE Error: ', rmsle(Y_dev, Y_dev_preds_ridge))

Y_dev_preds_ridgeCV = ridge_modelCV.predict(X_dev)
Y_dev_preds_ridgeCV = Y_dev_preds_ridgeCV.reshape(-1, 1)
print('RidgeCV model RMSE Error: ', rmsle(Y_dev, Y_dev_preds_ridgeCV))

ridge_preds = ridge_model.predict(X_test)
ridge_preds = np.expm1(ridge_preds)

ridgeCV_preds = ridge_modelCV.predict(X_test)
ridgeCV_preds = np.expm1(ridgeCV_preds)


def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):
    """
    對3個模型的預測值套用加權值，將3的預測值結合為1個預測值並傳回 
    Parameters: 
        Y1: 循環神經網路模型的預測值
        Y2: Ridge模型的預測值 
        Y3: RidgeCV模型的預測值 
        ratio1: 加權值 1 
        ratio2: 加權值 2
        
        (ratio3): 1.0 - ratio1 - ratio2
    """
    assert Y1.shape == Y2.shape
    return Y1 * ratio1 + Y2 * ratio2 + Y3 * (1 - ratio1 - ratio2)

best1 = 0
best2 = 0
lowest = 0.99
for i in range(100):
    for j in range(100):
        r = i * 0.01
        r2 = j * 0.01
        if ( r + r2 ) < 1.0:
            Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn,
                                              Y_dev_preds_ridge,
                                              Y_dev_preds_ridgeCV,
                                              r,
                                              r2)
            fpred = rmsle(Y_dev, Y_dev_preds)
            if fpred < lowest:
                best1 = r
                best2 = r2
                lowest = fpred

Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn,
                                  Y_dev_preds_ridge,
                                  Y_dev_preds_ridgeCV,
                                  best1,
                                  best2)
print('r1: ', best1)
print('r2: ', best2)
print('r3: ', 1.0 - best1 - best2)
print("(Best) RMSE error for RNN - Ridge - RidgeCV on dev set:\n", rmsle(Y_dev, Y_dev_preds))


preds = aggregate_predicts3(rnn_preds,
                            ridge_preds,
                            ridgeCV_preds,
                            best1,
                            best2)
submission = pd.DataFrame({"test_id": test_df.test_id,
                           "price": preds.reshape(-1)})

submission.to_csv("./rnn_ridge_submisson_best.csv", index = False)

stop_real = datetime.now()
execution_time_real = stop_real - start_real
print(execution_time_real)

