import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=1))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='mse', # categorical_crossentropy, binary_crossentropy, mse
              optimizer=tf.train.AdamOptimizer(0.001),
              metrics=['accuracy'])

def getBatch():
    x_batch = np.zeros((100,1))
    y_batch = np.zeros((100,2))
    for j in range(50):
        x_batch[2*j  ,:] = np.random.normal(loc=0.0, scale=1.0, size=(1,1))
        x_batch[2*j+1,:] = np.random.normal(loc=3.0, scale=1.0, size=(1,1))
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