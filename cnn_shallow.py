from scipy import misc
import numpy as np
import tensorflow as tf
import random

ntrain = 1000 # per class
ntest = 100 # per class
nclass = 2 # number of classes
vsize = 784 # vector size
batchsize = 10

Train = np.zeros((ntrain*nclass,vsize))
Test = np.zeros((ntest*nclass,vsize))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'MNIST/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        im = np.reshape(im,(1,-1)) # 1 by vsize
        itrain += 1
        Train[itrain,:] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'MNIST/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        im = np.reshape(im,(1,-1)) # 1 by vsize
        itest += 1
        Test[itest,:] = im
        LTest[itest,iclass] = 1 # 1-hot lable

x = tf.placeholder(tf.float32, [None, vsize])

W = tf.Variable(tf.zeros([vsize, nclass]))
b = tf.Variable(tf.zeros([nclass]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, nclass])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batch_xs = np.zeros((batchsize,vsize))
batch_ys = np.zeros((batchsize,nclass))
nsamples = ntrain*nclass
for i in range(1000):
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:] = Train[perm[j],:]
        batch_ys[j,:] = LTrain[perm[j],:]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: Test, y_: LTest}))