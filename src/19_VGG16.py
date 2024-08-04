import pandas as pd
import os, zipfile
from sklearn.model_selection import train_test_split

def prepare_data():
    """ 
    讀入資料，並區分訓練資料與驗證資料
    Returns:
        train_df(DataFrame)   : 從 train 取出用於訓練的資料 (90%)
        validate_df(DataFrame): 從 train 取出用於驗證的資料 (10%)
    """
    data = ['train', 'test']  # 設置資料集名稱列表
    path = './dogs-vs-cats-redux-kernels-edition/'  # 資料集路徑

    # 解壓縮資料集
    for el in data:
        with zipfile.ZipFile(path + el + ".zip", "r") as z:
            z.extractall(".")           # 將資料集解壓縮到當前目錄

    filenames = os.listdir("./train/")  # 獲取訓練資料夾中的所有文件名稱
    categories = []                     # 初始化類別列表
    for filename in filenames:
        category = filename.split('.')[0]   # 根據文件名分割得到類別名稱
        if category == 'dog':
            categories.append(1)            # 如果是狗，類別設置為 1
        else:
            categories.append(0)            # 如果是貓，類別設置為 0
    
    # 創建 DataFrame，包含文件名和類別
    df = pd.DataFrame({'filename': filenames,
                       'category': categories})
    
    # 將 DataFrame 分割為訓練集和驗證集，比例為 90% 和 10%
    train_df, validate_df = train_test_split(df, test_size=0.1)
    train_df = train_df.reset_index()           # 重置索引
    validate_df = validate_df.reset_index()     # 重置索引
    return train_df, validate_df                # 返回訓練集和驗證集

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def ImageDataGenrate(train_df, validate_df):
    """
    對圖像進行加工
    parameters:
        train_df(DataFrame)   : 從 train 取出用於訓練的資料 (90%)
        validate_df(DataFrame): 從 train 取出用於驗證的資料 (10%)
    Returns: 
        train(DirectoryIterator)     : 加工過後的訓練資料 
        valid(DirectoryIterator): 加工過後的驗證資料 
    """ 
    img_width, img_height = 224, 224        # 設置圖像寬度和高度
    target_size = (img_width, img_height)   # 設置目標尺寸
    batch_size = 16                         # 設置批次大小
    x_col, y_col = 'filename', 'category'   # 設置圖像文件名和類別列
    class_mode = 'binary'                   # 設置類別模式為二分類

    # 創建訓練數據增強生成器
    train_datagen = ImageDataGenerator(rotation_range=15,       # 旋轉範圍
                                       rescale=1. / 255,        # 縮放範圍
                                       shear_range=0.2,         # 剪切範圍
                                       zoom_range=0.2,          # 縮放範圍
                                       horizontal_flip=True,    # 水平翻轉
                                       fill_mode='nearest',     # 填充模式
                                       width_shift_range=0.1,   # 寬度平移範圍
                                       height_shift_range=0.1)  # 高度平移範圍
    
    train_df['category'] = train_df['category'].astype(str)             # 將類別轉為字符串類型
    train = train_datagen.flow_from_dataframe(train_df,                 # 使用訓練數據增強生成器生成訓練資料
                                              "./train/",               # 訓練資料目錄
                                              x_col=x_col,              # 設置圖像文件名列
                                              y_col=y_col,              # 設置類別列
                                              class_mode=class_mode,    # 設置類別模式
                                              target_size=target_size,  # 設置目標尺寸
                                              batch_size=batch_size,    # 設置批次大小
                                              shuffle=False)            # 不打亂數據順序
    
    # 創建驗證數據增強生成器
    valid_datagen = ImageDataGenerator(rescale=1. / 255)            # 驗證數據只進行縮放處理
    validate_df['category'] = validate_df['category'].astype(str)   # 將類別轉為字符串類型

    valid = valid_datagen.flow_from_dataframe(validate_df,              # 使用驗證數據增強生成器生成驗證資料
                                              "./train/",               # 訓練資料目錄
                                              x_col=x_col,              # 設置圖像文件名列
                                              y_col=y_col,              # 設置類別列
                                              class_mode=class_mode,    # 設置類別模式
                                              target_size=target_size,  # 設置目標尺寸
                                              batch_size=batch_size,    # 設置批次大小
                                              shuffle=False)            # 不打亂數據順序
    return train, valid                                                 # 返回訓練資料和驗證資料

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import LearningRateScheduler
import math

def train_FClayer(train_generator, validation_generator):
    """
    使用完成微調的 VGG16 進行訓練
    Returns: history(History 物件)
    """
    image_size = len(train_generator[0][0][0])                  # 獲取圖像大小
    input_shape = (image_size, image_size, 3)                   # 設置輸入形狀
    batch_size = len(train_generator[0][0])                     # 獲取批次大小
    total_train = len(train_generator) * batch_size             # 計算總訓練樣本數
    total_validate = len(validation_generator) * batch_size     # 計算總驗證樣本數

    # 加載預訓練的 VGG16 模型，不包含頂層
    pre_trained_model = VGG16(include_top=False,
                              weights='imagenet',
                              input_shape=input_shape)
    
    # 凍结前 15 層，訓練後面的層
    for layer in pre_trained_model.layers[:15]:
        layer.trainable = False
    for layer in pre_trained_model.layers[15:]:
        layer.trainable = True
        
    # 創建新的序列模型
    model = Sequential()
    model.add(pre_trained_model)        # 添加預訓練的 VGG16 模型
    model.add(GlobalMaxPooling2D())     # 添加全局最大池化層

    model.add(Dense(512, activation='relu'))    # 添加全連接層，512 個神經元，激活函數為 ReLU
    model.add(Dropout(0.5))                     # 添加 Dropout 層，防止過擬合，丟棄 50% 的神經元

    model.add(Dense(1, activation='sigmoid'))   # 添加輸出層，1 個神經元，激活函數為 sigmoid

    # 編譯模型，損失函數為二元交叉熵，優化器為 RMSprop，評估指標為準確率
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1e-5),
                  metrics=['accuracy'])
    model.summary()  # 顯示模型摘要

    def step_decay(epoch):
        """
        學習率衰減函數
        """
        initial_lrate = 0.00001     # 初始學習率
        drop = 0.5                  # 衰減率
        epochs_drop = 10.0          # 每多少個周期衰減一次
        lrate = initial_lrate * math.pow(drop, math.floor((epoch) / epochs_drop))  # 計算當前學習率
        return lrate
    
    lrate = LearningRateScheduler(step_decay)  # 創建學習率調度器

    epochs = 40                                                             # 訓練周期設置為 40
    history = model.fit(train_generator,                                    # 使用訓練生成器進行訓練
                        epochs=epochs,                                      # 訓練周期數
                        validation_data=validation_generator,               # 驗證數據集
                        validation_steps=total_validate // batch_size,      # 每個周期的驗證步數
                        steps_per_epoch=total_train // batch_size,          # 每個周期的訓練步數
                        verbose=1,                                          # 顯示訓練過程的進度條
                        callbacks=[lrate])                                  # 使用學習率調度器回調函數
    return history  # 返回訓練歷史

# 實際應用
train_df, validate_df = prepare_data()                  # 準備數據
train, valid = ImageDataGenrate(train_df, validate_df)  # 生成訓練資料和驗證資料
history = train_FClayer(train, valid)                   # 訓練模型
