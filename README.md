# classification-T108360150


1.Kaggle比賽簡介：
請訓練一個能辨認50個The Simpsons影集角色的 Convolutional Neural Network (CNN)。

評分:
以50個角色的平均辨認正確率（CategoryAccuracy）為準。
最差是隨便亂猜，此時CategoryAccuracy = 0.05
最好是全對：CategoryAccuracy = 1.00

2.檔案說明：

"basic-keras.ipynb"是訓練測試的ipynb檔

"IMG"中是報告需要的所有圖片

"model.h5"是訓練後的權重

"theSimpsons-test"中儲存了test的圖片
![image](https://github.com/108360150-Qiuchonghao/MachineLearning_Simpsons/blob/main/IMG/4.png)

"theSimpsons-train"中儲存了train的圖片
訓練集中每個角色有1000-2000張圖片，一共有50個角色。
![image](https://github.com/108360150-Qiuchonghao/MachineLearning_Simpsons/blob/main/IMG/5.png)

"keras_sub.csv"要submission的結果



3.程式：

我使用 jupyter notebook 編譯程式
![image](https://github.com/108360150-Qiuchonghao/MachineLearning_Houseprice/blob/main/IMG/jupyter1.jpg)


# Data Preprocessing：

In [1]:
      
        #import 所有所需要的API
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        import pickle
        import h5py
        import glob
        import time
        from random import shuffle
        from collections import Counter
        from sklearn.model_selection import train_test_split
        import keras
        from keras.preprocessing.image import ImageDataGenerator
        from keras.callbacks import LearningRateScheduler, ModelCheckpoint
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        from tensorflow.keras.optimizers import SGD, Adam
        from keras.utils import np_utils

In [2]:        
        # 卡通角色的Label-encoding
       map_characters = 
       {0: 'abraham_grampa_simpson', 1: 'agnes_skinner', 2: 'apu_nahasapeemapetilon', 
        3: 'barney_gumble', 4: 'bart_simpson', 5: 'brandine_spuckler', 6: 'carl_carlson', 
        7: 'charles_montgomery_burns', 8: 'chief_wiggum', 9: 'cletus_spuckler', 10: 'comic_book_guy', 
        11: 'disco_stu', 12: 'dolph_starbeam', 13: 'duff_man', 14: 'edna_krabappel', 15: 'fat_tony', 
        16: 'gary_chalmers', 17: 'gil',18: 'groundskeeper_willie' , 19: 'homer_simpson', 20: 'jimbo_jones', 
        21: 'kearney_zzyzwicz', 22: 'kent_brockman', 23: 'krusty_the_clown', 24: 'lenny_leonard', 25: 'lionel_hutz', 
        26: 'lisa_simpson', 27: 'lunchlady_doris',28:'maggie_simpson',29:'marge_simpson',30:'martin_prince',
        31:'mayor_quimby',32:'milhouse_van_houten',
        33:'miss_hoover',34:'moe_szyslak',
        35:'ned_flanders',36:'nelson_muntz',37:'otto_mann',38:'patty_bouvier',39:'principal_skinner',
        40:'professor_john_frink',41:'rainier_wolfcastle',42:'ralph_wiggum',43:'selma_bouvier',44:'sideshow_bob',
        45:'sideshow_mel',46:'snake_jailbird',47:'timothy_lovejoy',48:'troy_mcclure', 49:'waylon_smithers'}

        img_width = 42 
        img_height = 42

        num_classes = len(map_characters) # 要辨識的角色種類
        pictures_per_class = 2100 # 每個角色會有接近2000張訓練圖像
        test_size = 0.15
        imgsPath = "/home/qiu/study_college/machine_learning/Simpsons/machine-learningntut-2021-autumn-classification/homework/classification-T108360150/theSimpsons-train/train"



In [3]:              

        def load_pictures():
            pics = []
            labels = []
            
            for k, v in map_characters.items(): # k: 數字編碼 v: 角色label
                # 把某一個角色在檔案夾裡的所有圖像檔的路徑捉出來
                pictures = [k for k in glob.glob(imgsPath + "/" + v + "/*")]        
                print(v + " : " + str(len(pictures))) # 看一下每個角色有多少訓練圖像
                for i, pic in enumerate(pictures):
                    tmp_img = cv2.imread(pic)
                    
                    # 由於OpenCv讀圖像時是以BGR (Blue-Green-Red), 我們把它轉置成RGB (Red-Green-Blue)
                    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
                    tmp_img = cv2.resize(tmp_img, (img_height, img_width)) # 進行大小歸一位            
                    pics.append(tmp_img)
                    labels.append(k)    
            return np.array(pics), np.array(labels)

        # 取得訓練資料集與驗證資料集
        def get_dataset(save=False, load=False):
            if load: 
                # 從檔案系統中載入之前處理保存的訓練資料集與驗證資料集
                h5f = h5py.File('dataset.h5','r')
                X_train = h5f['X_train'][:]
                X_test = h5f['X_test'][:]
                h5f.close()
                
                # 從檔案系統中載入之前處理保存的訓練資料標籤與驗證資料集籤
                h5f = h5py.File('labels.h5', 'r')
                y_train = h5f['y_train'][:]
                y_test = h5f['y_test'][:]
                h5f.close()
            else:
                # 從最原始的圖像檔案開始處理
                X, y = load_pictures()
                y = np_utils.to_categorical(y, num_classes) # 目標的類別種類數
                
                # 將資料切分為訓練資料集與驗證資料集 (85% vs. 15%)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) 
                if save: # 保存尚未進行歸一化的圖像數據
                    h5f = h5py.File('dataset.h5', 'w')
                    h5f.create_dataset('X_train', data=X_train)
                    h5f.create_dataset('X_test', data=X_test)
                    h5f.close()
                    
                    h5f = h5py.File('labels.h5', 'w')
                    h5f.create_dataset('y_train', data=y_train)
                    h5f.create_dataset('y_test', data=y_test)
                    h5f.close()
            
            # 進行圖像每個像素值的型別轉換與歸一化處理
            X_train = X_train.astype('float32') / 255.
            X_test = X_test.astype('float32') / 255.
            print("Train", X_train.shape, y_train.shape)
            print("Test", X_test.shape, y_test.shape)
            
            return X_train, X_test, y_train, y_test   

取得訓練資料集與驗證資料集  

In [4]:      

        # 取得訓練資料集與驗證資料集  
         X_train, X_test, y_train, y_test = get_dataset(save=True, load=False)
![image](https://github.com/108360150-Qiuchonghao/MachineLearning_Simpsons/blob/main/IMG/2.png)


現在我們來定義我們的模型架構。我們將使用具有6個卷積層的前饋網絡，然後是完全連接的隱藏層。
我們也將在兩者之間使用Dropout層來防止網絡"過擬合(overfitting)"。

In [5]:      

        def create_model_six_conv(input_shape):
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            
            model.add(Dense(num_classes, activation='softmax'))
            
            return model;
![image](https://github.com/108360150-Qiuchonghao/MachineLearning_Simpsons/blob/main/IMG/3.png)


在訓練模型之前，我們需要將模型配置為學習算法並進行編譯。我們需要指定:
loss: 損失函數，我們要優化。我們不能使用MSE，因為它是不連續的數值。因此，我們使用：categorical_crossentropy
optimizer: 我們使用標準隨機梯度下降(Stochastic gradient descent)與聶士傑洛夫動量(Nesterov momentum)
metric: 由於我們正在處理一個分類問題，我們用度量是accuracy。

In [6]:      

        # 讓我們先配置一個常用的組合來作為後續優化的基準點
        lr = 0.01
        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# Training PART：
In [7]:      

        def lr_schedule(epoch):
            return lr
        #*(0.1**int(epoch/10))

        batch_size = 128
        epochs = 250

        history = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                shuffle=True,
                callbacks=[LearningRateScheduler(lr_schedule),
                    ModelCheckpoint('model.h5', save_best_only=True)
                ])

In [8]:         

        #透過趨勢圖來觀察訓練與驗證的走向 ，是否有overfitting。
        import matplotlib.pyplot as plt

        def plot_train_history(history, train_metrics, val_metrics):
            plt.plot(history.history.get(train_metrics),'-o')
            plt.plot(history.history.get(val_metrics),'-o')
            plt.ylabel(train_metrics)
            plt.xlabel('Epochs')
            plt.legend(['train', 'validation'])
            
            
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plot_train_history(history, 'loss','val_loss')

        plt.subplot(1,2,2)
        plot_train_history(history, 'acc','val_acc')

        plt.show()

Epochs200-250的訓練結果如圖所示：
![image](https://github.com/108360150-Qiuchonghao/MachineLearning_Simpsons/blob/main/IMG/1.png)

# Test：
In [9]:   

        # 預測與比對
        #將預測結果存到“keras_sub.csv”中
        from keras.models import load_model
        def read_images(path):
            images=[]
            for i in range(10791):
                image = cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(img_height,img_width))
                image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
                images.append(image)
            images = np.array(images,dtype = np.float32)/255
            return images
            
        imgsPath = "/home/qiu/study_college/machine_learning/Simpsons/machine-learningntut-2021-autumn-classification/homework/classification-T108360150/theSimpsons-test"    
        # 把訓練時val_loss最小的模型載入
        #model = load_model('model-dtaug.h5')
        model = load_model('model.h5')

        test_images=read_images(imgsPath)
        pred =model.predict(test_images)
        pred = np.argmax(pred,axis = 1)
        with open('keras_sub.csv','w')as f:
                                    f.write('id,character\n')
                                    for i in range(len(pred)):
                                            f.write(str(i+1)+','+map_characters[pred[i]]+'\n')



4.心得：
這次的比賽相比於房價預測更加的有趣，也讓我們感覺到其實用keras做影像的識別並沒有特別的困難。

在對圖片的預處理中，第一步是調整它們的大小。我們需要有相同大小的所有照片進行訓練。
我會將數據資料轉換型別為float32來節省一些ram用量並對它們進行歸一化（除以255）。
然後，我用數字代替每一個卡通每一個角色的名稱，並感謝Keras，我可以快速地將這些類別裝換為向量：

這次比賽train模型的時間非常的長，因為我使用cpu作為工具，跑250個Epoch後模型接近overfitting，用時將近3小時，若改用GPU進行運算，時間會大幅度提升。

![image](https://github.com/108360150-Qiuchonghao/MachineLearning_Simpsons/blob/main/IMG/6.png)