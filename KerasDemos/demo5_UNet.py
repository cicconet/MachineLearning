import numpy as np
import tensorflow as tf
import os, shutil, sys

from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, concatenate
from tensorflow.keras import Input, Model

import skimage
from matplotlib import pyplot as plt
def imshow(I):
    skimage.io.imshow(I)
    plt.show()

def cat1(I,J):
    return np.concatenate((I,J),axis=1)

# --------------------------------------------------
# setup
# --------------------------------------------------

imSize = 64
nClasses = 2
nChannels = 1
batchSize = 8

# --------------------------------------------------
# data
# --------------------------------------------------

def getImLb():
    x,y = np.meshgrid(np.arange(imSize),np.arange(imSize))
    x0 = np.random.randint(imSize)
    y0 = np.random.randint(imSize)
    r = imSize/4
    M = np.sqrt((x-x0)**2+(y-y0)**2) < r
    I = 0.5+0.25*np.cos(2*np.pi*x/imSize)
    N = np.random.normal(size=(imSize,imSize))
    I[M] = 0.25+0.5*np.random.rand()
    I = I+0.01*N
    I = I-np.min(I)
    I = I/np.max(I)
    L = np.zeros((imSize,imSize,nClasses))
    L[:,:,0] = M
    nf = np.sum(M)
    nb = imSize**2-nf
    B = np.random.rand(imSize,imSize) < nf/nb
    B[M] = 0
    L[:,:,1] = B
    return I,L

def getBatch(n):
    x_batch = np.zeros((n,imSize,imSize,nChannels))
    y_batch = np.zeros((n,imSize,imSize,nClasses))
    for i in range(n):
        I,L = getImLb()
        x_batch[i,:,:,0], y_batch[i,:,:,:] = I, L
        # imshow(np.concatenate((I,np.concatenate((L[:,:,0],L[:,:,1]),axis=1)),axis=1))
    return x_batch, y_batch

# I,L = getImLb()
# imshow(np.concatenate((I,np.concatenate((L[:,:,0],L[:,:,1]),axis=1)),axis=1))
# sys.exit(0)

# --------------------------------------------------
# model
# --------------------------------------------------

x = Input((imSize,imSize,nChannels))

nFeatMaps = [16,32,64,128]

hidden = [x]
for i in range(len(nFeatMaps)-1):
    hidden.append(Conv2D(nFeatMaps[i],(3,3),padding='same',activation='relu')(hidden[-1]))
    hidden.append(MaxPooling2D()(hidden[-1]))

hidden.append(Conv2D(nFeatMaps[-1],(3,3),padding='same',activation='relu')(hidden[-1]))

for i in range(len(nFeatMaps)-1):
    hidden.append(Conv2DTranspose(nFeatMaps[-i-2],(3,3),strides=(2,2),padding='same',activation='relu')(hidden[-1]))
    hidden.append(concatenate([hidden[-1], hidden[2*(len(nFeatMaps)-1)-1-2*i]]))
    hidden.append(Conv2D(nFeatMaps[-i-2],(3,3),padding='same',activation='relu')(hidden[-1]))

hidden.append(Conv2D(nClasses,(1,1),padding='same',activation='softmax')(hidden[-1]))

model = Model(hidden[0],hidden[-1])

def custom_acc(y_true, y_pred):
    l = []
    for iClass in range(nClasses):
        labels0 = tf.reshape(tf.to_int32(tf.slice(y_true,[0,0,0,iClass],[-1,-1,-1,1])),[batchSize,imSize,imSize])
        predict0 = tf.reshape(tf.to_int32(tf.equal(tf.argmax(y_pred,3),iClass)),[batchSize,imSize,imSize])
        correct = tf.multiply(labels0,predict0)
        nCorrect0 = tf.reduce_sum(correct)
        nLabels0 = tf.reduce_sum(labels0)
        l.append(tf.to_float(nCorrect0)/tf.to_float(nLabels0))
    return tf.add_n(l)/nClasses

def custom_loss(y_true, y_pred):
    return -tf.reduce_sum(tf.multiply(y_true,tf.log(y_pred)))

model.compile(loss=custom_loss, optimizer=tf.train.AdamOptimizer(0.001), metrics=[custom_acc])

# built in cross entropy and accuracy do not handle unlabeled pixels
# model.compile(loss='categorical_crossentropy',optimizer=tf.train.AdamOptimizer(0.001),metrics=['accuracy'])

model.summary()

# --------------------------------------------------
# train
# --------------------------------------------------

modelPath = '/home/cicconet/Workspace/TFModel/Keras'
# model.load_weights(modelPath)

logDir = '/home/cicconet/Workspace/TFLog/Keras'
if os.path.exists(logDir):
    shutil.rmtree(logDir)
writer = tf.summary.FileWriter(logDir)

tags = ['loss','acc']
for i in range(1000):
    x_batch, y_batch = getBatch(batchSize)
    model.train_on_batch(x_batch, y_batch)

    if i % 10 == 0:
        x_batch, y_batch = getBatch(batchSize)
        loss_and_metrics = model.evaluate(x_batch, y_batch, batch_size=batchSize, verbose=0)
        print('step: %d, loss: %f, acc: %f' % (i,loss_and_metrics[0],loss_and_metrics[1]))

        for j in range(2):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tags[j], simple_value=loss_and_metrics[j])])
            writer.add_summary(summary,i)

x,_ = getBatch(10)
y = model.predict(x)
for i in range(10):
    print('predict', i)
    skimage.io.imsave('/home/cicconet/Workspace/Scratch/I%05d.png' % i, np.uint8(255*cat1(x[i,:,:,0],y[i,:,:,0])))

writer.close()
model.save_weights(modelPath)