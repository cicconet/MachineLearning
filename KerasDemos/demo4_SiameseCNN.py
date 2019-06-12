import sys

import numpy as np
from skimage.io import imread, imsave

import tensorflow as tf
from tensorflow.keras.layers import Dense, subtract, Lambda, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input, Model

import os, shutil

imSize = 28
nChannels = 1

inputX = Input(batch_shape=(None,imSize,imSize,nChannels))

conv = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(imSize,imSize,nChannels))(inputX)
maxp = MaxPooling2D(pool_size=(2, 2))(conv)
flat = Flatten()(maxp)
fc = Dense(128, activation='relu')(flat)
repres = Model(inputX,fc)

repres.summary()


inputA = Input(batch_shape=(None,imSize,imSize,nChannels))
inputB = Input(batch_shape=(None,imSize,imSize,nChannels))

represA = repres(inputA)
represB = repres(inputB)

merged = subtract([represA, represB])
absMerged = Lambda(lambda x: tf.abs(x))(merged)


pred = Dense(2, activation='softmax')(absMerged)
siam = Model([inputA, inputB], pred)

siam.compile(loss='categorical_crossentropy', # categorical_crossentropy, binary_crossentropy, mse
             optimizer=tf.train.AdamOptimizer(0.001),
             metrics=['accuracy'])
siam.summary()

modelDir = '/home/cicconet/Workspace/TFModel/Keras'
# siam.load_weights(modelDir)

logDir = '/home/cicconet/Workspace/TFLog/Keras'
if os.path.exists(logDir):
    shutil.rmtree(logDir)
writer = tf.summary.FileWriter(logDir)

def getBatch(n):
    a_batch = np.zeros((n,imSize,imSize,nChannels))
    b_batch = np.zeros((n,imSize,imSize,nChannels))
    y_batch = np.zeros((n,2))
    for j in range(int(n/2)):
        # same class
        digit = np.random.randint(10)
        indA = np.random.randint(1000)
        indB = np.random.randint(1000)
        imA = imread('/home/cicconet/Documents/Python/MachineLearning/MNIST/Train/%d/Image%05d.png' % (digit,indA)).astype('double')/255
        imB = imread('/home/cicconet/Documents/Python/MachineLearning/MNIST/Train/%d/Image%05d.png' % (digit,indB)).astype('double')/255

        a_batch[2*j,:,:,0] = imA
        b_batch[2*j,:,:,0] = imB
        y_batch[2*j,0] = 1

        # diff classes
        digit1 = np.random.randint(10)
        digit2 = digit1
        while digit2 == digit1:
            digit2 = np.random.randint(10)
        indA = np.random.randint(1000)
        indB = np.random.randint(1000)
        imA = imread('/home/cicconet/Documents/Python/MachineLearning/MNIST/Train/%d/Image%05d.png' % (digit1,indA)).astype('double')/255
        imB = imread('/home/cicconet/Documents/Python/MachineLearning/MNIST/Train/%d/Image%05d.png' % (digit2,indB)).astype('double')/255

        a_batch[2*j+1,:,:,0] = imA
        b_batch[2*j+1,:,:,0] = imB
        y_batch[2*j+1,1] = 1

    return a_batch, b_batch, y_batch


tags = ['loss','acc']
for i in range(1000):
    a_batch, b_batch, y_batch = getBatch(100)
    siam.train_on_batch([a_batch,b_batch], y_batch)

    if i % 10 == 0:
        a_test, b_test, y_test = getBatch(100)
        loss_and_metrics = siam.evaluate([a_test,b_test], y_test, batch_size=10, verbose=0)
        print('step: %d, loss: %f, acc: %f' % (i,loss_and_metrics[0],loss_and_metrics[1]))

        for j in range(2):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tags[j], simple_value=loss_and_metrics[j])])
            writer.add_summary(summary,i)
writer.close()
siam.save_weights(modelDir)


a_test, b_test, y_test = getBatch(10)
probs = siam.predict([a_test, b_test], batch_size=10)
c = np.argmax(probs,axis=1)
for i in range(len(c)):
    im1 = np.uint8(255*a_test[i])
    im2 = np.uint8(255*b_test[i])
    imsave('/home/cicconet/Workspace/Scratch/I%05d_%d.png' % (i,c[i]), np.concatenate((im1,im2),axis=1)[:,:,0])
    print('%d: %d, (%f, %f)' % (i, c[i], probs[i,0], probs[i,1]))

loss_and_metrics = siam.evaluate([a_test,b_test], y_test, batch_size=10, verbose=0)
print('test acc: %f' % loss_and_metrics[1])