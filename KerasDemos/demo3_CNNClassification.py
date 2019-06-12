import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input, Model

from skimage.io import imread, imsave

import os, shutil

imSize = 28
nChannels = 1
nClasses = 4
batchSize = 64
nImagesPerClass = 500

x = Input(batch_shape=(None,imSize,imSize,nChannels))

conv = Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(imSize,imSize,nChannels))(x)
maxp = MaxPooling2D(pool_size=(2, 2))(conv)
conv = Conv2D(32, kernel_size=(3, 3), activation='relu')(maxp)
maxp = MaxPooling2D(pool_size=(2, 2))(conv)

flat = Flatten()(maxp)
fc = Dense(1024, activation='relu')(flat)
sm = Dense(nClasses, activation='softmax')(fc)
model = Model(x,sm)

model.summary()

model.compile(loss='categorical_crossentropy', # categorical_crossentropy, binary_crossentropy, mse
              optimizer=tf.train.AdamOptimizer(0.001),
              metrics=['accuracy'])

def getBatch():
    x_batch = np.zeros((batchSize,imSize,imSize,nChannels))
    y_batch = np.zeros((batchSize,nClasses))
    for j in range(batchSize):
        digit = np.random.randint(nClasses)
        index = np.random.randint(nImagesPerClass)
        image = imread('/home/cicconet/Documents/Python/MachineLearning/Cells/Train/%d/Image%05d.png' % (digit,index)).astype('double')/255
        x_batch[j,:,:,0] = image
        y_batch[j,digit] = 1
    return x_batch, y_batch

logDir = '/home/cicconet/Workspace/TFLog/Keras'
if os.path.exists(logDir):
    shutil.rmtree(logDir)
writer = tf.summary.FileWriter(logDir)

modelDir = '/home/cicconet/Workspace/TFModel/Keras'
# model.load_weights(modelDir)

tags = ['loss','acc']
for i in range(1000):
    x_batch, y_batch = getBatch()
    model.train_on_batch(x_batch, y_batch)

    if i % 10 == 0:
        x_test, y_test = getBatch()
        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
        print('step: %d, loss: %f, acc: %f' % (i,loss_and_metrics[0],loss_and_metrics[1]))

        for j in range(2):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tags[j], simple_value=loss_and_metrics[j])])
            writer.add_summary(summary,i)

writer.close()
model.save_weights(modelDir)

x_test, y_test = getBatch()

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
print('test acc: %f' % loss_and_metrics[1])