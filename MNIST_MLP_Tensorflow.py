
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


# In[2]:

mnist = input_data.read_data_sets("MNIST_data/",one_hot =True)


# In[95]:

it=[]
acadam=[]



#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer().minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if _%100==0:
        it.append(_)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        acadam.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


# In[99]:

plt.plot(it, ac, color='red', label='GD')
plt.plot(it,acmo,color='peru',label='Momentum')
plt.plot(it,acada,color='green',label='Adagrade')
plt.plot(it,acadam,color='blue',label='Adam')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[20]:

#GD
it=[]
gd3=[]



#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(3).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if _%100==0:
        it.append(_)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #here
        gd3.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


# In[21]:

plt.plot(it,gd01, color='red', label='GD lr=0.1')
plt.plot(it,gd001,color='peru',label='GD lr=0.01')
plt.plot(it,gd0001,color='green',label='GD lr=0.001')
plt.plot(it,gd1,color='blue',label='GD lr=1')
plt.plot(it,gd3,color='m',label='GD lr=3')
plt.plot(it,gd5,color='aqua',label='GD lr=5')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[33]:

#Adam
it=[]
ad0003=[]



#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(0.003).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if _%100==0:
        it.append(_)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #here
        ad0003.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


# In[45]:

plt.plot(it,ad0003,color='m',label='Adam lr=0.0003')
plt.plot(it,ad0001,color='peru',label='Adam lr=0.001')
plt.plot(it,ad001, color='red', label='Adam lr=0.01')
plt.plot(it,ad003,color='blue',label='Adam lr=0.03')
plt.plot(it,ad01,color='green',label='Adam lr=0.1')


#plt.plot(it,gd5,color='aqua',label='GD lr=5')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[47]:

#adagrade
it=[]
adg1=[]



#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step=tf.train.AdagradOptimizer(1).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if _%100==0:
        it.append(_)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #here
        adg1.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


# In[49]:

plt.plot(it,adg001,color='peru',label='Adag lr=0.01')
plt.plot(it,adg01, color='red', label='Adag lr=0.1')

plt.plot(it,adg03,color='green',label='Adag lr=0.3')
plt.plot(it,adg05,color='blue',label='Adag lr=0.5')
plt.plot(it,adg1,color='m',label='Adag lr=1')
#plt.plot(it,gd5,color='aqua',label='GD lr=5')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[65]:

#momentum
it=[]
mm1_05=[]



#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step=tf.train.MomentumOptimizer(1,0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if _%100==0:
        it.append(_)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #here
        mm1_05.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


# In[66]:

plt.plot(it,mm001_05,color='peru',label='lr=0.01 g =0.5')
plt.plot(it,mm01_05,color='red',label='lr=0.1 g =0.5')
plt.plot(it,mm01_03,color='green',label='lr=0.1 g =0.3')
plt.plot(it,mm01_01,color='blue',label='lr=0.1 g =0.1')
plt.plot(it,mm01_1,color='m',label='lr=0.1 g =1')
plt.plot(it,mm1_05,color='aqua',label='lr=1 g =0.5')
#plt.plot(it,gd5,color='aqua',label='GD lr=5')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[85]:

#RMS prop
it=[]
rmsprop0005_09_00=[]



#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step=tf.train.RMSPropOptimizer(learning_rate=0.005,decay=0.9,momentum=0.0).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if _%100==0:
        it.append(_)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #here
        rmsprop0005_09_00.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


# In[87]:

plt.plot(it,rmsprop001_09_00,color='peru',label='lr=0.01 decay =0.9')
plt.plot(it,rmsprop0001_09_00,color='red',label='lr=0.001 decay =0.9')
plt.plot(it,rmsprop0003_09_00,color='green',label='lr=0.003 decay =0.9')
plt.plot(it,rmsprop0005_09_00,color='blue',label='lr=0.005 decay =0.9')

plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[101]:

it=[]
rms=[]
rmst=[]


#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step=tf.train.RMSPropOptimizer(learning_rate=0.005,decay=0.9,momentum=0.0).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if _%100==0:
        it.append(_)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        rms.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
        rmst.append(accuracy.eval({x:mnist.train.images,y_:mnist.train.labels}))


# In[100]:

plt.plot(it, adgt, color='m', label='Adagradtrain')
plt.plot(it,adg,color='aqua',label='Adagradetest')
plt.plot(it, mmt, color='red', label='Momentumtrain')
plt.plot(it,mm,color='peru',label='Momentumtest')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[102]:

plt.plot(it,acadamt,color='green',label='Adamtrain')
plt.plot(it,acadam,color='blue',label='Adamtest')
plt.plot(it, gdt, color='red', label='GDtrain')
plt.plot(it,gd,color='peru',label='GDtest')
plt.plot(it, rmst, color='m', label='RMSProptrain')
plt.plot(it,rms,color='aqua',label='RMSProptest')

plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[3]:

it=[]
acadam=[]



#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.reduce_sum((y_-y)**2))
train_step=tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#sess = tf.InteractiveSession()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        train_step.run({x: batch_xs, y_: batch_ys})
        if _%100==0:
            it.append(_)
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

            acadam.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


# In[4]:

#plt.plot(it, ac, color='red', label='GD')
#plt.plot(it,acmo,color='peru',label='Momentum')
#plt.plot(it,acada,color='green',label='Adagrade')
plt.plot(it,acadam,color='blue',label='Adam')
#plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()


# In[18]:

import numpy as np
np.shape(batch_xs)


# In[73]:

np.shape(batch_xs)


# In[46]:

aaa = [4,6,8,10]


# In[58]:

tf.reduce_mean(tf.cast(tf.equal(tf.argmax(aaa,1),tf.argmax(aaa,1)),tf.float32)).eval


# In[65]:

import keras


# In[66]:

import tensorflow as tf

with tf.name_scope('hidden') as scope:
    a = tf.constant(5, name='alpha')
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
    b = tf.Variable(tf.zeros([1]), name='biases')


# In[ ]:




# In[89]:

it=[]
#acadam=[]
acmomen = []


#ac
in_units = 784
h1_units = 300
x=tf.placeholder(tf.float32,[None,in_units])
W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))
lr= tf.constant(0.01)
hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.nn.softmax(tf.matmul(hidden1,W2)+b2)
y_ = tf.placeholder(tf.float32,[None,10])
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.reduce_sum((y_-y)**2))
train_step=tf.train.MomentumOptimizer(lr,0.5).minimize(cross_entropy)
#sess = tf.InteractiveSession()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        learning_rate = 0.01
        if _>300:#从第20个iteration开始考虑自动优化器
            learning_rate=0.001
        #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        train_step.run({x: batch_xs, y_: batch_ys,lr:learning_rate})
#         learning_rate = 1
#         if _>20:#从第20个iteration开始考虑自动优化器
#             learning_rate=0.1
        
        if _%100==0:
            it.append(_)
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

            acmomen.append(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))


# In[90]:

#plt.plot(it, ac, color='red', label='GD')
plt.plot(it,acmomen,color='peru',label='Momentum0.1;0.5')
#plt.plot(it,acada,color='green',label='Adagrade')
plt.plot(it,acadam,color='blue',label='Adam0.01')
plt.legend(loc='lower right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show() 


# In[ ]:



