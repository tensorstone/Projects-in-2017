
# coding: utf-8

# In[1]:

import os
import glob
os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

os.chdir("/home/ai-i-sunhao/CIFAR/train")

import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import csv


# In[2]:

f=open('/home/ai-i-sunhao/trainLabels.csv')
reader = csv.reader(f)
ID = []
label = []
for row in reader:
    ID.append(row[0])
    label.append(row[1])


# In[3]:

for i in range(len(label)-1):
    ID[i] = ID[i+1]
    label[i] = label[i+1]
ID = ID[0:50000]
label = label[0:50000]
numlabel = label
    


# In[4]:

for i in range(len(label)):
    if label[i]=='airplane':
        numlabel[i]=0
    if label[i]=='automobile':
        numlabel[i]=1
    if label[i]=='bird':
        numlabel[i]=2
    if label[i]=='cat':
        numlabel[i]=3
    if label[i]=='deer':
        numlabel[i]=4
    if label[i]=='dog':
        numlabel[i]=5
    if label[i]=='frog':
        numlabel[i]=6
    if label[i]=='horse':
        numlabel[i]=7
    if label[i]=='ship':
        numlabel[i]=8
    if label[i]=='truck':
        numlabel[i]=9


# In[5]:

for i in range(len(ID)):
    ID[i] = np.int(ID[i])
    numlabel[i] = np.int(label[i])


# In[6]:

import os
from PIL import Image
import numpy as np
#os.chdir('/Users/sunhop/Desktop/')
imgs = 0
def load_data():
    data = np.empty((50000,32,32,3),dtype="float32")
    label = np.empty((50000,),dtype="uint8")
    for i in range(50000):
        imgs= glob.glob("/home/ai-i-sunhao/CIFAR/train/" + str(i+1) +".png")
        img = Image.open(imgs[0])
        arr = np.asarray(img,dtype = "float32")
        data[i,:,:,:] = arr #arr_t
        label[i] = numlabel[i]
    data =data/255
    #data =data- np.mean(data)
    return data,label


# In[7]:

data ,label = load_data()


# In[66]:

from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
#from data import load_data
import random
import numpy as np

np.random.seed(1024)  # for reproducibility

#加载数据
data,label = load_data()

#打乱数据


index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
print(data.shape[0], ' samples')

#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, 10)

###############kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)
#开始建立CNN模型
###############


# In[11]:

data = np.reshape(data,[-1,32,3,32])


# In[13]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(3,3),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(32,3,32))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[8]:

data=np.reshape(data,[-1,3,32,32])


# In[9]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(3,3),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(3,32,32))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[22]:

data= np.reshape(data,[-1,32,32,3])


# In[17]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(3,3),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(32,32,3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[ ]:




# In[ ]:




# In[26]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
data = np.reshape(data,[-1,32*32*3])
model = Sequential()

L = 32

model.add(Dense( 512,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(32*32*3,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('relu'))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[16]:

data = np.reshape(data,[-1,32,32,3])
car = data[2]
plt.imshow(car)
plt.show()


# In[18]:

testindex = [i for i in range(32)]
index_shuf = [i for i in range(32)]
random.shuffle(index_shuf)
index_reverse = [i for i in reversed(testindex)]


# In[28]:

car.shape
reverse_car_col = car[index_reverse,:,:]
reverse_car_vol = car[:,index_reverse,:]
reverse_car_vol_cal = reverse_car_vol[index_reverse,:,:]
shuf_car_col = car[index_shuf,:,:]



# In[31]:

ind = []
for i in range(8):
    ind.extend([i for i in reversed([4*i,4*i+1,4*i+2,4*i+3])])
ind = np.asarray(ind)
ind


# In[32]:

plt.imshow(car[ind,:,:])
plt.show()


# In[33]:

ind = []
for i in range(16):
    ind.extend([i for i in reversed([2*i,2*i+1])])
ind = np.asarray(ind)
plt.imshow(car[ind,:,:])
plt.show()


# In[34]:

for i in range(len(data)):
    data[i] = data[i][ind,:,:]


# In[37]:

plt.imshow(data[2])
plt.show()


# In[38]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(3,3),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(32,32,3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[53]:

ind = []
for i in range(8):
    ind.extend([i for i in reversed([4*i,4*i+1,4*i+2,4*i+3])])
ind = np.asarray(ind)


for i in range(len(data)):
    data[i] = data[i][ind,:,:]


# In[54]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(3,3),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(32,32,3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[59]:

ind = []
for i in range(8):
    ind.extend([i for i in reversed([4*i,4*i+1,4*i+2,4*i+3])])
ind = np.asarray(ind)


# In[64]:

for i in range(8):
    temp = ind[4*i:4*i+4]
    random.shuffle(temp)


# In[65]:

ind


# In[67]:

for i in range(len(data)):
    data[i] = data[i][ind,:,:]


# In[68]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(3,3),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(32,32,3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[69]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(5,5),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(32,32,3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[71]:

#生成一个model
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization

model = Sequential()

L = 32

model.add(Conv2D( 32 ,(8,8),padding='same',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0),input_shape=(32,32,3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


model.add(Conv2D(32, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3),kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense( 512, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=0)))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.0065, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

tensorboard = TensorBoard(log_dir='./logs/run_BN', histogram_freq=0)
checkpoint = ModelCheckpoint('model_CIFAR_run1.h5',monitor = 'val_acc',verbose = 1,save_best_only = True)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

#model.fit(data, label, batch_size=100, nb_epoch=60,shuffle=True,verbose=1,validation_split=0.2)
model.fit(data, label, batch_size=50, nb_epoch=300,shuffle=True,verbose=1,validation_split=0.2, callbacks=[checkpoint,tensorboard,EarlyStopping])


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[11]:

import os
from PIL import Image
import numpy as np
#os.chdir('/Users/sunhop/Desktop/')
imgs = 0
def load_test_data():
    data = np.empty((300000,32,32,3),dtype="float32")
    #label = np.empty((50000,),dtype="uint8")
    for i in range(300000):
        imgs= glob.glob("/home/ai-i-sunhao/CIFAR/test/" + str(i+1) +".png")
        img = Image.open(imgs[0])
        arr = np.asarray(img,dtype = "float32")
        data[i,:,:,:] = arr #arr_t
        #label[i] = numlabel[i]
    data =data/255
    #data =data- np.mean(data)
    return data


# In[12]:

xtest = load_test_data()


# In[14]:

xtest.shape


# In[16]:

yy = model.predict_classes(xtest,batch_size=32,verbose=1)


# In[22]:

len(yy)
np.savetxt('temp.csv',yy,delimiter = ',')

