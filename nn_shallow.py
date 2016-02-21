from scipy import misc
import numpy as np
import tensorflow as tf
import random

# --------------------------------------------------
# setup

ntrain = 1000 # per class
ntest = 100 # per class
nclass = 10 # number of classes
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
        path = '/Users/Cicconet/MacDev/TensorFlow/MNIST/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        im = np.reshape(im,(1,-1)) # 1 by vsize
        itrain += 1
        Train[itrain,:] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = '/Users/Cicconet/MacDev/TensorFlow/MNIST/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        im = np.reshape(im,(1,-1)) # 1 by vsize
        itest += 1
        Test[itest,:] = im
        LTest[itest,iclass] = 1 # 1-hot lable

tf_data = tf.placeholder(tf.float32, [None, vsize])
tf_labels = tf.placeholder(tf.float32, [None, nclass])

# --------------------------------------------------
# model

W = tf.Variable(tf.zeros([vsize, nclass]))
b = tf.Variable(tf.zeros([nclass]))

forward = tf.nn.softmax(tf.matmul(tf_data, W) + b)

# --------------------------------------------------
# loss

cross_entropy = -tf.reduce_sum(tf_labels*tf.log(forward))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

evaluation = tf.equal(tf.argmax(forward,1), tf.argmax(tf_labels,1))
accuracy = tf.reduce_mean(tf.cast(evaluation, "float"))

# --------------------------------------------------
# optimization

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
    sess.run(optimizer, feed_dict={tf_data: batch_xs, tf_labels: batch_ys})

# --------------------------------------------------
# test

print(sess.run(accuracy, feed_dict={tf_data: Test, tf_labels: LTest}))