import numpy as np
import tensorflow as tf
import os, shutil, sys

from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, concatenate, BatchNormalization
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
t = tf.placeholder(tf.bool)

nFeatMaps = [16,32,64,128]

hidden = [x]
ccidx = [] # index of downsampling layers to concatenate
for i in range(len(nFeatMaps)-1):
    hidden.append(Conv2D(nFeatMaps[i],(3,3),padding='same',activation='relu')(hidden[-1]))
    hidden.append(Conv2D(nFeatMaps[i],(3,3),padding='same',activation='relu')(hidden[-1]))
    ccidx.append(len(hidden)-1)
    hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))
    hidden.append(MaxPooling2D()(hidden[-1]))

hidden.append(Conv2D(nFeatMaps[-1],(3,3),padding='same',activation='relu')(hidden[-1]))
hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))

for i in range(len(nFeatMaps)-1):
    hidden.append(Conv2DTranspose(nFeatMaps[-i-2],(3,3),strides=(2,2),padding='same',activation='relu')(hidden[-1]))
    hidden.append(concatenate([hidden[-1], hidden[ccidx[-1-i]]]))
    hidden.append(Conv2D(nFeatMaps[-i-2],(3,3),padding='same',activation='relu')(hidden[-1]))
    hidden.append(Conv2D(nFeatMaps[-i-2],(3,3),padding='same',activation='relu')(hidden[-1]))
    hidden.append(tf.layers.batch_normalization(hidden[-1], training=t))

hidden.append(Conv2D(nClasses,(1,1),padding='same',activation='softmax')(hidden[-1]))

# model = Model(hidden[0],hidden[-1])
# model.summary()
# sys.exit(0)

sm = hidden[-1]
y = Input((imSize,imSize,nClasses))

l = []
for iClass in range(nClasses):
    labels0 = tf.reshape(tf.to_int32(tf.slice(y,[0,0,0,iClass],[-1,-1,-1,1])),[batchSize,imSize,imSize])
    predict0 = tf.reshape(tf.to_int32(tf.equal(tf.argmax(sm,3),iClass)),[batchSize,imSize,imSize])
    correct = tf.multiply(labels0,predict0)
    nCorrect0 = tf.reduce_sum(correct)
    nLabels0 = tf.reduce_sum(labels0)
    l.append(tf.to_float(nCorrect0)/tf.to_float(nLabels0))
acc = tf.add_n(l)/nClasses

loss = -tf.reduce_sum(tf.multiply(y,tf.log(sm)))
updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(0.001)
with tf.control_dependencies(updateOps):
    optOp = optimizer.minimize(loss)

tf.summary.scalar('loss', loss)
tf.summary.scalar('acc', acc)
planeIm = tf.slice(x,[0,0,0,0],[-1,-1,-1,1])
planeImN = tf.div(tf.subtract(planeIm,tf.reduce_min(planeIm,axis=(1,2),keep_dims=True)),tf.subtract(tf.reduce_max(planeIm,axis=(1,2),keep_dims=True),tf.reduce_min(planeIm,axis=(1,2),keep_dims=True)))
pmIndex = 0
planePM = tf.slice(sm,[0,0,0,pmIndex],[-1,-1,-1,1])
plane = tf.concat([planeImN,planePM],2)
tf.summary.image('impm',plane,max_outputs=4)
mergedsmr = tf.summary.merge_all()

# --------------------------------------------------
# train
# --------------------------------------------------

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) # config parameter needed to save variables when using GPU

modelPath = '/home/cicconet/Workspace/TFModel/Keras'

logDir = '/home/cicconet/Workspace/TFLog/Keras'
if os.path.exists(logDir):
    shutil.rmtree(logDir)
writer = tf.summary.FileWriter(logDir)

restoreVariables = False
if restoreVariables:
    saver.restore(sess, modelPath)
    print('Model restored.')
else:
    sess.run(tf.global_variables_initializer())

for i in range(100):
    x_batch, y_batch = getBatch(batchSize)
    summary,a,_ = sess.run([mergedsmr,acc,optOp],feed_dict={x: x_batch, y: y_batch, t: True})
    writer.add_summary(summary, i)
    print(i,a)

print('Model saved in file: %s' % saver.save(sess, modelPath))

x_batch,_ = getBatch(10)
y_pred = sess.run(sm,feed_dict={x: x_batch, t: False}) # in theory t should be False during testing but sometimes True works better
for i in range(10):
    print('predict', i)
    skimage.io.imsave('/home/cicconet/Workspace/Scratch/I%05d.png' % i, np.uint8(255*cat1(x_batch[i,:,:,0],y_pred[i,:,:,0])))

writer.close()
sess.close()