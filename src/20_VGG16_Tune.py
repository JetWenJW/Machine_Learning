import pandas as pd
import os, zipfile
from sklearn.model_selection import train_test_split

def prepareData():
    """ 
    讀入資料，並區分訓練資料與驗證資料
    Returns:
        train_df(DataFrame)   :從train取出用於訓練的資料 (90%)
        validate_df(DataFrame):從train取出用於驗證的資料 (10%)
    """
    # 將訓練資料跟測試資料解壓縮
    # 解壓縮的zip檔名
    data = ['train', 'test']

    # 在當前的目錄解壓縮train.zip、test.zip
    path = './dogs-vs-cats-redux-kernels-edition/'
    for el in data:
        with zipfile.ZipFile(path + el + ".zip", "r") as z:
            z.extractall(".")

    # 使用檔名dog.x.jpg、cat.x.jpg，建立標籤1與0 
    # 取得train資料夾內的檔名，放入filenames
    filenames = os.listdir("./train")
    # 放置標籤的清單
    categories = [] 
    for filename in filenames: 
        # 分割檔名、取出最前頭的元素(dog/cat)
        # 將dog為1、cat為0設為標籤，放入category  
        category = filename.split('.')[0] 
        if category == 'dog': # 若為 dog，則加上標籤1  
            categories.append(1) 
        else: # 若為cat，則加上標籤0
            categories.append(0)

    # 對df的列filename放入檔名filename
    # 對列category放入標籤數值categories
    df = pd.DataFrame({'filename': filenames,
                       'category': categories})

    # 將訓練資料總數25000已隨機方式分割為90%跟10%、
    # 90%為用於訓練的資料、10%為用於驗證的資料
    train_df, validate_df = train_test_split(df, test_size=0.1)
    # 重新配置列的索引
    train_df = train_df.reset_index()
    validate_df = validate_df.reset_index()

    return train_df, validate_df



from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 從Keras導入ImageDataGenerator，用於圖像數據的增強和預處理

def ImageDataGenerate(train_df, validate_df):
    """
    對圖像進行加工
    parameters:
        train_df(DataFrame)   : 從train取出用於訓練的資料(90%) 
        validate_df(DataFrame): 從train取出用於驗證的資料(10%) 
    Returns: 
        train_generator(DirectoryIterator)     : 加工過後的訓練資料 
        validation_generator(DirectoryIterator): 加工過後的驗證資料 
    """ 
    # 重新調整圖像尺寸
    img_width, img_height = 224, 224            # 設定目標圖像的寬度和高度
    target_size = (img_width, img_height)       # 定義圖像的目標尺寸
    # 批次大小
    batch_size = 16                             # 設定每個批次的圖像數量

    # 檔名的欄位名稱，標籤的欄位名稱
    x_col, y_col = 'filename', 'category'       # 設定DataFrame中存儲檔名和標籤的欄位名稱
    # 設定flow_from_dataframe()的class_mode數值 
    # 此範例為二元分類，設定值為'binary'
    class_mode = 'binary'                       # 因為是二元分類問題，class_mode設為'binary'

    # 建立Generator來加工圖像
    train_datagen = ImageDataGenerator(
        rotation_range=15,                      # 隨機旋轉圖像的角度範圍，15度內
        rescale=1./255,                         # 將圖像像素值縮放到[0, 1]範圍
        shear_range=0.2,                        # 隨機剪切的角度範圍
        zoom_range=0.2,                         # 隨機縮放的範圍
        horizontal_flip=True,                   # 隨機水平翻轉圖像
        fill_mode='nearest',                    # 填充新出現的圖像區域，使用最接近的像素值
        width_shift_range=0.1,                  # 隨機水平位移的範圍
        height_shift_range=0.1                  # 隨機垂直位移的範圍
    )

    # 因為沒有輸出層，故class_mode為None
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,             # 輸入的訓練數據的DataFrame
        directory="./train/",           # 圖像文件所在的目錄
        x_col=x_col,                    # DataFrame中存儲圖像檔名的欄位名稱
        y_col=y_col,                    # DataFrame中存儲標籤的欄位名稱
        class_mode=None,                # 不需要生成標籤（僅圖像生成），所以設為None
        target_size=target_size,        # 將圖像調整為目標尺寸
        batch_size=batch_size,          # 設定批次大小
        shuffle=False                   # 不打亂圖像順序（通常在預測時使用）
    )

    # 使用Generator產生加工完的驗證資料
    valid_datagen = ImageDataGenerator(
        rescale=1./255                      # 將圖像像素值縮放到[0, 1]範圍
    )

    # 使用Generator產生預處理的驗證資料
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=validate_df,              # 輸入的驗證數據的DataFrame
        directory="./train/",               # 圖像文件所在的目錄
        x_col=x_col,                        # DataFrame中存儲圖像檔名的欄位名稱
        y_col=y_col,                        # DataFrame中存儲標籤的欄位名稱
        class_mode=None,                    # 不需要生成標籤（僅圖像生成），所以設為None
        target_size=target_size,            # 將圖像調整為目標尺寸
        batch_size=batch_size,              # 設定批次大小
        shuffle=False                       # 不打亂圖像順序（通常在預測時使用）
    )

    # 傳回訓練資料與驗證資料
    return train_generator, valid_generator  # 返回訓練和驗證數據生成器



from tensorflow.keras.applications import VGG16     # 從TensorFlow的Keras應用中導入VGG16模型
import numpy as np                                  # 導入NumPy，用於數值計算和處理

def save_VGG16_outputs(train, valid):
    '''
    將訓練資料、驗證資料輸入VGG16
    並將兩者的輸出儲存為 npy檔

    parameters:
    train(DataFrameIterator): 預處理完成的訓練資料
    valid(DataFrameIterator): 預處理完成的驗證資料
    '''
    # 取得圖像尺寸
    image_size = len(train[0][0][0])            # 從訓練資料生成器的第一個批次中獲取圖像的尺寸
    # 將輸入資料的形狀改為Tuple
    input_shape = (image_size, image_size, 3)   # 定義VGG16模型所需的輸入形狀，這裡假設圖像大小一致

    # 讀入VGG16模型與預學習之參數
    model = VGG16(include_top = False,          # 不包含VGG16模型的全連接層
                  weights = 'imagenet',         # 使用在ImageNet上預訓練的權重
                  input_shape = input_shape)    # 定義模型的輸入形狀
    #顯示 VGG16概要
    model.summary()  # 輸出模型的架構摘要

    # 將訓練資料輸入VGG16模型
    vgg16_train = model.predict(train,                  # 使用訓練數據生成器進行預測
                                steps = len(train),     # 設定步數為生成器的總步數
                                verbose = 1)            # 設定日誌輸出詳情

    # 儲存訓練資料的輸出結果
    np.save('vgg16_train.npy', vgg16_train)             # 將訓練資料的預測結果儲存為NumPy檔案

    # 將驗證資料輸入VGG16模型
    vgg16_test = model.predict(valid,                   # 使用驗證數據生成器進行預測
                               steps = len(valid),      # 設定步數為生成器的總步數
                               verbose = 1)             # 設定日誌輸出詳情

    # 儲存驗證資料的輸出結果
    np.save('vgg16_test.npy', vgg16_test)               # 將驗證資料的預測結果儲存為NumPy檔案


import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout, GlobalMaxPooling2D, Dense

def train_FClayer(train_labels, validation_labels):
    '''
    將VGG16的輸出放入自創的FC層進行學習
    parameters:
        train_labels(int的list)   : 訓練資料的正確答案標籤
        validate_labels(int的list): 驗證資料的正確答案標籤
    '''
    # 將VGG16的訓練資料輸出讀入NumPy序列
    train_data = np.load('vgg16_train.npy')
    # 將VGG16的驗證資料輸出讀入NumPy序列
    validation_data = np.load('vgg16_test.npy')
    
    # 製作自創的神經網路結構 
    model = Sequential() 
    # 對四維張量(batch_size, rows, cols, channels)套用池化演算法後
    # 拉平為二維張量(batch_size, channels) 
    model.add(GlobalMaxPooling2D()) 
    # 全連接層 
    model.add(Dense(512,                # 神經元數為 512
                    activation='relu')) # 激活函數為 ReLU
    # 丟棄率50%
    model.add(Dropout(0.5))

    # 輸出層
    model.add(Dense(1,                     # 神經元數為 1
                    activation='sigmoid')) # 激活函數為Sigmoid

    # 模型編譯
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate = 1e-5), # 學習率為預設值的1/100
                  metrics=['accuracy'])

    # 訓練模型
    epoch = 20      # 訓練週期
    batch_size = 16 # 批次大小
    history = model.fit(train_data,   # 訓練資料
                        train_labels, # 訓練資料正確答案
                        epochs=epoch,
                        batch_size=batch_size,
                        verbose=1,
                        # 驗證資料與正確答案
                        validation_data=(validation_data,
                                         validation_labels))

    # 傳回history
    return history


train_df, validate_df = prepareData()
train, valid = ImageDataGenerate(train_df, validate_df)
save_VGG16_outputs(train, valid)

train_labels = np.array(train_df['category'])
# 取得驗證資料的正確答案
validation_labels = np.array(validate_df['category'])
# 執行訓練模型
history = train_FClayer(train_labels,validation_labels)

