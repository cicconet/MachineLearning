import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input, Model


imSize = 28
nChannels = 1

x = Input(batch_shape=(None,imSize,imSize,nChannels))

conv = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(imSize,imSize,nChannels))(x)
maxp = MaxPooling2D(pool_size=(2, 2))(conv)
flat = Flatten()(maxp)
fc = Dense(128, activation='relu')(flat)
sm = Dense(2, activation='softmax')(fc)
model = Model(x,sm)

model.summary()

model.compile(loss='categorical_crossentropy', # categorical_crossentropy, binary_crossentropy, mse
              optimizer=tf.train.AdamOptimizer(0.001),
              metrics=['accuracy'])

def getBatch():
    x_batch = np.zeros((32,imSize,imSize,nChannels))
    y_batch = np.zeros((32,2))
    for j in range(16):
        x_batch[2*j  ,:,:,:] = np.random.normal(loc=0.0, scale=1.0, size=(imSize,imSize,nChannels))
        x_batch[2*j+1,:,:,:] = np.random.normal(loc=0.0, scale=2.0, size=(imSize,imSize,nChannels))
        y_batch[2*j  ,0] = 1
        y_batch[2*j+1,1] = 1
    return x_batch, y_batch

for i in range(1000):
    x_batch, y_batch = getBatch()
    model.train_on_batch(x_batch, y_batch)

    if i % 100 == 0:
        x_test, y_test = getBatch()
        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
        print('step: %d, loss: %f, acc: %f' % (i,loss_and_metrics[0],loss_and_metrics[1]))


x_test, y_test = getBatch()

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
print('test acc: %f' % loss_and_metrics[1])