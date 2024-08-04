import numpy as np              # 導入NumPy，用於數學運算
import pandas as pd             # 導入Pandas，用於數據處理
import os                       # 導入OS，用於操作系統接口

# 列出 ./dogs-vs-cats-redux-kernels-edition 目錄下的所有文件
for dirname, _, filenames in os.walk('./dogs-vs-cats-redux-kernels-edition'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import shutil, zipfile                          # 導入shutil和zipfile，用於文件操作和解壓縮
data = ['train', 'test']                        # 定義數據集文件名
path = './dogs-vs-cats-redux-kernels-edition/'  # 定義數據集路徑

# 解壓縮數據集
for el in data:
    zip_file_path = path + el + '.zip'              # 定義每個壓縮包的完整路徑
    extract_path = path + el                        # 定義解壓縮後的目標路徑
    with zipfile.ZipFile(zip_file_path, 'r') as z:  # 打開壓縮包
        z.extractall(extract_path)                  # 解壓縮到指定目錄
    print(f"{el}數據集解壓縮完成")                  # 打印解壓縮完成訊息

print("所有數據集解壓縮完成")                       # 打印最終完成訊息

# 獲取訓練數據文件名列表
filenames = os.listdir("./train")
categories = []  # 初始化類別列表

# 根據文件名前綴確定類別
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)  # 狗為1類
    else:
        categories.append(0)  # 貓為0類

# 創建包含文件名和類別的DataFrame
df = pd.DataFrame({'filename': filenames,
                   'category': categories})

# 顯示數據框架的前五行
df.head()

# 繪製類別分佈的條形圖
df['category'].value_counts().plot.bar()

from tensorflow.keras.preprocessing.image import load_img   # 導入load_img，用於加載圖像
import matplotlib.pyplot as plt                             # 導入matplotlib.pyplot，用於繪圖
import random                                               # 導入random，用於隨機選擇樣本

# 隨機選擇16張圖片
sample = random.sample(filenames, 16)

# 設置圖像大小和佈局
plt.figure(figsize = (12, 12))

# 繪製隨機選擇的16張圖片
for i in range(0, 16):
    plt.subplot(4, 4, i + 1)
    fname = sample[i]
    image = load_img("./train/" + fname)
    plt.imshow(image)
    plt.axis('off')
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split  # 導入train_test_split，用於劃分訓練和驗證數據

# 劃分訓練和驗證數據
train_df, validate_df = train_test_split(df, test_size=0.1)

# 重置訓練和驗證數據索引
train_df = train_df.reset_index()
validate_df = validate_df.reset_index()

# 獲取訓練和驗證樣本數
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

# 輸出訓練和驗證樣本數
print(total_train)
print(total_validate)

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 導入ImageDataGenerator，用於圖像數據增強

# 定義圖像寬度和高度
img_width, img_height = 224, 224
target_size = (img_width, img_height)   # 定義目標尺寸
batch_size = 16                         # 定義批次大小
x_col, y_col = 'filename', 'category'   # 定義特徵列和標籤列
class_mode = 'binary'                   # 定義分類模式為二分類

# 創建訓練數據增強生成器
train_datagen = ImageDataGenerator(rescale = 1. / 255,
                                   rotation_range = 15,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1)

train_df['category'] = train_df['category'].astype(str)  # 將訓練數據的類別轉換為字符串

train_generator = train_datagen.flow_from_dataframe(
    train_df,                                   # 包含文件名和類別標籤的 DataFrame
    "./train/",                                 # 圖像文件所在的目錄
    x_col=x_col,                                # DataFrame 中包含圖像文件名的列名稱
    y_col=y_col,                                # DataFrame 中包含圖像類別的列名稱
    class_mode=class_mode,                      # 定義返回的標籤類型
    target_size=target_size,                    # 生成的圖像的目標大小 (height, width)
    batch_size=batch_size                       # 每個批次生成的圖像數量
)

# 創建驗證數據增強生成器
valid_datagen = ImageDataGenerator(rescale = 1. / 255)
validate_df['category'] = validate_df['category'].astype(str)  # 將驗證數據的類別轉換為字符串

# 創建驗證數據生成器
valid_generator = valid_datagen.flow_from_dataframe(validate_df,
                                                    "./train/",
                                                    x_col=x_col,
                                                    y_col=y_col,
                                                    class_mode=class_mode,
                                                    target_size=target_size,
                                                    batch_size=batch_size)

# 隨機選擇一個訓練數據樣本
example_df = train_df.sample(n=1).reset_index(drop=True)

# 創建示例數據生成器
example_generator = train_datagen.flow_from_dataframe(example_df,
                                                      "./train/",
                                                      x_col='filename',
                                                      y_col='category',
                                                      target_size=target_size)

# 繪製數據增強後的示例圖片
plt.figure(figsize=(12, 12))
for i in range(0, 9):
    plt.subplot(3, 3, i + 1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.show()

from tensorflow.keras.models import Sequential                                      # 導入Sequential，用於創建序列模型
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense   # 導入層，用於構建模型
from tensorflow.keras.layers import GlobalMaxPooling2D          # 導入GlobalMaxPooling2D，用於全局最大池化
from tensorflow.keras import optimizers                         # 導入優化器
from tensorflow.keras import regularizers                       # 導入正則化器

# 創建序列模型
model = Sequential()
input_shape = (img_width, img_height, 3)  # 定義輸入形狀

# 添加卷積層、池化層和Dropout層
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128,
                 kernel_size = (3, 3),
                 padding = 'same',
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# 添加全局最大池化層
model.add(GlobalMaxPooling2D())

# 添加全連接層和Dropout層
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))

# 添加輸出層
model.add(Dense(1, activation = 'sigmoid'))

# 編譯模型
model.compile(loss = 'binary_crossentropy',
              metrics = ['accuracy'],
              optimizer = optimizers.RMSprop())

import math  # 導入math，用於數學運算
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, Callback  # 導入回調函數

# 定義學習率調度函數
def step_decay(epoch):
    initial_lrate = 0.001   # 初始學習率設為0.001
    drop = 0.5              # 每次學習率衰減的比例設為0.5
    epochs_drop = 10.0      # 每10個epoch學習率衰減一次
    # 計算衰減後的學習率
    lrate = initial_lrate * math.pow(drop, math.floor((epoch) / epochs_drop))  
    return lrate            # 返回計算後的學習率

# 創建學習率調度器，使用step_decay函數作為調度策略
lrate = LearningRateScheduler(step_decay)

# 創建早停回調函數
earstop = EarlyStopping(monitor = 'val_loss',     # 監控驗證集損失
                        min_delta = 0,            # 最小變化幅度
                        patience = 5)             # 如果連續5個epoch驗證集損失不減小則停止訓練

# 設置訓練週期數
epochs = 40

# 訓練模型
history = model.fit(train_generator,                                  # 使用訓練數據生成器
                    epochs = epochs,                                  # 設置訓練的epoch數
                    steps_per_epoch = total_train // batch_size,      # 每個epoch的步數
                    validation_data = valid_generator,                # 使用驗證數據生成器
                    validation_steps = total_validate // batch_size,  # 每個epoch的驗證步數
                    verbose = 1,                                      # 訓練過程中輸出進度條
                    callbacks = [lrate, earstop])                     # 設置回調函數，包括學習率調度器和早停回調函數
