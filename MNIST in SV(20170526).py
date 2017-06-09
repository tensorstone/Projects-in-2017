
# coding: utf-8

# In[3]:

import os
from PIL import Image
import numpy as np
#os.chdir('/Users/sunhop/Desktop/')
def load_data():
    data = np.empty((42000,1,28,28),dtype="float32")
    label = np.empty((42000,),dtype="uint8")
    
    imgs = os.listdir("/Users/sunhop/Desktop/mnist")
    num = len(imgs)
    for i in range(num):
        img = Image.open("/Users/sunhop/Desktop/mnist/"+imgs[i])
        arr = np.asarray(img,dtype = "float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    data =data/ np.max(data)
    data =data- np.mean(data)
    return data,label

import pandas as pd
HWtraindata = pd.read_csv("MNISTtrain.csv",header=None)
def load_kaggle_data():
    data = np.empty((42000,1,28,28),dtype = "float32")
    label = np.empty((42000,),dtype = "uint8")
    lenth = 42000
    for i in range(lenth):
        for j in range(28):
            data[i,0,j,:] = HWtraindata.values[i][1+j*28:1+j*28+28]
        label[i] = HWtraindata.values[i][0]
    data = data / np.max(data)
    data = data - np.mean(data)
    return data,label

HWtestdata = pd.read_csv("MNISTtest.csv")
def load_kaggle_test():
    lenth = len(HWtestdata.values)
    test = np.empty((lenth,1,28,28),dtype = "float32")
    for i in range(lenth):
        for j in range(28):
            test[i,0,j,:] = HWtestdata.values[i][j*28:j*28+28]
    test = test / np.max(test)
    test = test - np.mean(test)
    return test
def load_expanded_kaggle_data():
    L = 42000;
    N = 3
    data = np.empty((L*N,1,28,28),dtype = "float32")
    label = np.empty((L*N,),dtype = "uint8")
    lenth = L*N
    for i in range(L):
        for j in range(28):
            data[i,0,j,:] = HWtraindata.values[i][1+j*28:1+j*28+28]
        label[i] = HWtraindata.values[i][0]
    for i in range(L):
        for j in range(27):
            data[i+L,0,j,:] = HWtraindata.values[i][3+j*28:3+j*28+28]
        for j in range(27,28):
            data[i+L,0,j,:] = HWtraindata.values[i][1+j*28:1+j*28+28]
        label[i+L] = HWtraindata.values[i][0]
    for i in range(L):
        for j in range(1):
            data[i+2*L,0,j,:] = HWtraindata.values[i][1+j*28:1+j*28+28]
        for j in range(1,28):
            data[i+2*L,0,j,:] = HWtraindata.values[i][j*28-1:j*28 + 28 - 1]
        label[i+2*L] = HWtraindata.values[i][0]
    data = data / np.max(data)
    data = data - np.mean(data)
    return data,label
def load_expanded_expanded_kaggle_data():
    L = 42000;
    N = 5
    data = np.empty((L*N,1,28,28),dtype = "float32")
    label = np.empty((L*N,),dtype = "uint8")
    lenth = L*N
    for i in range(L):
        for j in range(28):
            data[i,0,j,:] = HWtraindata.values[i][1+j*28:1+j*28+28]
        label[i] = HWtraindata.values[i][0]
    for i in range(L):
        for j in range(27):
            data[i+L,0,j,:] = HWtraindata.values[i][3+j*28:3+j*28+28]
        for j in range(27,28):
            data[i+L,0,j,:] = HWtraindata.values[i][1+j*28:1+j*28+28]
        label[i+L] = HWtraindata.values[i][0]#向右平移
    for i in range(L):
        for j in range(1):
            data[i+2*L,0,j,:] = HWtraindata.values[i][1:29]
        for j in range(1,28):
            data[i+2*L,0,j,:] = HWtraindata.values[i][j*28-1:j*28 + 28 - 1]
        label[i+2*L] = HWtraindata.values[i][0]#向左平移
        
    for i in range(L):
        for j in range(1):
            data[i+3*L,0,j,:] = HWtraindata.values[i][756:784]
        for j in range(1,28):
            data[i+3*L,0,j,:] = HWtraindata.values[i][(j-1)*28+1:(j-1)*28 + 28 +1]
        label[i+3*L] = HWtraindata.values[i][0]#向上平移
    for i in range(L):
        for j in range(27):
            data[i+4*L,0,j,:] = HWtraindata.values[i][(j+1)*28+1:(j+1)*28+28+1]
        for j in range(27,28):
            data[i+4*L,0,j,:] = HWtraindata.values[i][1:29]
        label[i+4*L] = HWtraindata.values[i][0]#向下平移
    
    data = data / np.max(data)
    data = data - np.mean(data)
    return data,label


# In[4]:



'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py

2016.06.06更新：
这份代码是keras开发初期写的，当时keras还没有现在这么流行，文档也还没那么丰富，所以我当时写了一些简单的教程。
现在keras的API也发生了一些的变化，建议及推荐直接上keras.io看更加详细的教程。

'''
#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
#from data import load_data
import random
import numpy as np

np.random.seed(1024)  # for reproducibility

#加载数据
data, label = load_expanded_expanded_kaggle_data()
#打乱数据


index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
print(data.shape[0], ' samples')

#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, 10)

###############
#开始建立CNN模型
###############


# In[ ]:

#99.329#
model = Sequential()

L = 50

model.add(Convolution2D(L, 3, 3, border_mode='same',input_shape=(1,28,28))) 

#第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
#border_mode可以是valid或者full，具体看这里说明：http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
#激活函数用tanh
#你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))

model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))

#第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(2*L, 5, 5, border_mode='same'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))

#第三个卷积层，16个卷积核，每个卷积核大小3*3
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(4*L, 3, 3, border_mode='same')) 
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))

#全连接层，先将前一层输出的二维特征图flatten为一维的。
#Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4
#全连接有128个神经元节点,初始化方式为normal

#model.add(Convolution2D(8*L, 3, 3, border_mode='same')) 
#model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))


#model.add(Convolution2D(16*L, 3, 3, border_mode='same')) 
#model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))



model.add(Flatten())
model.add(Dense(800, init='normal'))
model.add(Activation('relu'))
#model.add(Dense(300, init='normal'))
#model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(Dense(500, init='normal'))
#model.add(Activation('relu'))

#Softmax分类，输出是10类别
model.add(Dense(10, init='normal'))
model.add(Activation('softmax'))
#model.add(Dropout(0.5))

##############
#开始训练模型
##############
#使用SGD + momentum
#model.compile里的参数loss就是损失函数(目标函数)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
#数据经过随机打乱shuffle=True。verbose=1，训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
#validation_split=0.2，将20%的数据作为验证集。
model.fit(data, label, batch_size=100, nb_epoch=200,shuffle=True,verbose=1,validation_split=0.05)


# In[ ]:



