import pandas as pd                                     # 導入pandas，用於數據處理
import os, zipfile                                      # 導入os和zipfile，用於文件和壓縮文件處理
from sklearn.model_selection import train_test_split    # 導入train_test_split，用於劃分訓練和驗證數據

def prepare_data():
    """ 
    讀入資料，並區分訓練資料與驗證資料
    Returns:
        train_df(DataFrame): 從train取出用於訓練的資料 (90%)
        validate_df(DataFrame): 從train取出用於驗證的資料 (10%)
    """
    data = ['train', 'test']                        # 定義需要解壓的數據集
    path = './dogs-vs-cats-redux-kernels-edition/'  # 定義數據集路徑

    for el in data:
        with zipfile.ZipFile(path + el +".zip", "r") as z:
            z.extractall(".")                       # 解壓縮數據集

    filenames = os.listdir("./train/")              # 獲取訓練數據文件名列表
    categories = []                                 # 初始化類別列表
    for filename in filenames:
        category = filename.split('.')[0]           # 根據文件名前綴確定類別
        if category == 'dog':
            categories.append(1)                    # 狗為1類
        else:
            categories.append(0)                    # 貓為0類

    df = pd.DataFrame({'filename': filenames, 'category': categories})  # 創建包含文件名和類別的DataFrame
    
    train_df, validate_df = train_test_split(df, test_size=0.1)         # 劃分訓練和驗證數據
    train_df = train_df.reset_index()                                   # 重置訓練數據索引
    validate_df = validate_df.reset_index()                             # 重置驗證數據索引
    return train_df, validate_df                                        # 返回訓練和驗證數據

from tensorflow.keras.preprocessing.image import ImageDataGenerator     # 導入ImageDataGenerator，用於圖像數據增強

def ImageDataGenrate(train_df, validate_df):
    """
    對圖像進行加工
    parameters:
        train_df(DataFrame): 從train取出用於訓練的資料 (90%)
        validate_df(DataFrame): 從train取出用於驗證的資料 (10%)
    Returns: 
        train(DirectoryIterator): 加工過後的訓練資料 
        valid(DirectoryIterator): 加工過後的驗證資料 
    """
    img_width, img_height = 224, 224        # 定義圖像的寬度和高度
    target_size = (img_width, img_height)   # 定義目標尺寸
    batch_size = 16                         # 定義批次大小
    x_col, y_col = 'filename', 'category'   # 定義特徵列和標籤列
    class_mode = 'binary'                   # 定義分類模式為二分類
    
    train_datagen = ImageDataGenerator(
        rotation_range=15,          # 旋轉範圍
        rescale=1. / 255,           # 將圖像像素值縮放到0~1之間
        shear_range=0.2,            # 剪切強度
        zoom_range=0.2,             # 縮放範圍
        horizontal_flip=True,       # 水平翻轉
        fill_mode='nearest',        # 填充模式
        width_shift_range=0.1,      # 水平位移範圍
        height_shift_range=0.1      # 垂直位移範圍
    )
    
    train_df['category'] = train_df['category'].astype(str)  # 將訓練數據的類別轉換為字符串
    train = train_datagen.flow_from_dataframe(
        train_df,                       # 輸入訓練數據DataFrame
        "./train/",                     # 訓練數據目錄
        x_col=x_col,                    # 特徵列
        y_col=y_col,                    # 標籤列
        class_mode=class_mode,          # 分類模式
        target_size=target_size,        # 圖像目標尺寸
        batch_size=batch_size,          # 批次大小
        shuffle=False                   # 不進行隨機打亂
    )
    
    valid_datagen = ImageDataGenerator(rescale=1. / 255)            # 只進行縮放操作的驗證數據增強
    validate_df['category'] = validate_df['category'].astype(str)   # 將驗證數據的類別轉換為字符串

    valid = valid_datagen.flow_from_dataframe(
        validate_df,                # 輸入驗證數據DataFrame
        "./train/",                 # 驗證數據目錄
        x_col=x_col,                # 特徵列
        y_col=y_col,                # 標籤列
        class_mode=class_mode,      # 分類模式
        target_size=target_size,    # 圖像目標尺寸
        batch_size=batch_size,      # 批次大小
        shuffle=False               # 不進行隨機打亂
    )
    return train, valid             # 返回加工過的訓練和驗證數據

from tensorflow.keras.models import Sequential                          # 導入Sequential，用於創建序列模型
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D  # 導入層，用於構建模型
from tensorflow.keras import optimizers                                 # 導入優化器
from tensorflow.keras.applications import VGG16                         # 導入VGG16預訓練模型
from tensorflow.keras.callbacks import LearningRateScheduler            # 導入LearningRateScheduler，用於調整學習率
import math  # 導入math，用於數學運算

def train_FClayer(train_generator, validation_generator):
    """
    使用完成微調的VGG16進行訓練
    Returns: history(History物件)
    """
    image_size = len(train_generator[0][0][0])                  # 獲取圖像尺寸
    input_shape = (image_size, image_size, 3)                   # 定義輸入形狀
    batch_size = len(train_generator[0][0])                     # 獲取批次大小
    total_train = len(train_generator) * batch_size             # 計算總訓練樣本數
    total_validate = len(validation_generator) * batch_size     # 計算總驗證樣本數

    pre_trained_model = VGG16(
        include_top=False,          # 不包含全連接層
        weights='imagenet',         # 使用ImageNet數據集的預訓練權重
        input_shape=input_shape     # 定義輸入形狀
    )

    for layer in pre_trained_model.layers[:15]:
        layer.trainable = False                 # 凍結前15層
    for layer in pre_trained_model.layers[15:]:
        layer.trainable = True                  # 解凍剩餘層
    
    model = Sequential()                        # 創建序列模型
    model.add(pre_trained_model)                # 添加預訓練模型
    model.add(GlobalMaxPooling2D())             # 添加全局最大池化層
    model.add(Dense(512, activation='relu'))    # 添加全連接層，使用ReLU激活函數
    model.add(Dropout(0.5))                     # 添加Dropout層，防止過擬合
    model.add(Dense(1, activation='sigmoid'))   # 添加輸出層，使用Sigmoid激活函數

    model.compile(
        loss='binary_crossentropy',                         # 使用二元交叉熵損失函數
        optimizer=optimizers.RMSprop(learning_rate=1e-5),   # 使用RMSprop優化器
        metrics=['accuracy']                                # 設置評估指標為準確率
    )

    model.summary()  # 輸出模型摘要

    def step_decay(epoch):
        initial_lrate = 0.00001     # 初始學習率
        drop = 0.5                  # 衰減率
        epochs_drop = 10.0          # 衰減週期
        lrate = initial_lrate * math.pow(drop, math.floor((epoch) / epochs_drop))  # 計算學習率
        return lrate
    
    lrate = LearningRateScheduler(step_decay)  # 創建學習率調度器

    epochs = 40             # 設置訓練週期數
    history = model.fit(
        train_generator,    # 訓練數據生成器
        epochs=epochs,      # 訓練週期數
        validation_data=validation_generator,               # 驗證數據生成器
        validation_steps=total_validate // batch_size,      # 驗證步數
        steps_per_epoch=total_train // batch_size,          # 每個訓練週期的步數
        verbose=1,                                          # 設置輸出詳情等級
        callbacks=[lrate]                                   # 設置回調函數
    )
    return history  # 返回訓練歷史

# 實際應用
train_df, validate_df = prepare_data()                  # 準備數據
train, valid = ImageDataGenrate(train_df, validate_df)  # 生成訓練和驗證數據
history = train_FClayer(train, valid)                   # 訓練模型
