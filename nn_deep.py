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

tf_data = tf.placeholder(tf.float32, [None, vsize])
tf_labels = tf.placeholder(tf.float32, [None, nclass])

# --------------------------------------------------
# model: 1 hidden layer

# nhidden_1 = 512;
# W_1 = tf.Variable(tf.truncated_normal([vsize, nhidden_1], stddev=0.1))
# b_1 = tf.Variable(tf.constant(0.1, shape=[nhidden_1]))
# out_layer_1 = tf.nn.relu(tf.matmul(tf_data, W_1) + b_1)

# W_2 = tf.Variable(tf.truncated_normal([nhidden_1, nclass], stddev=0.1))
# b_2 = tf.Variable(tf.constant(0.1, shape=[nclass]))
# out_layer_2 = tf.matmul(out_layer_1, W_2) + b_2

# forward = tf.nn.softmax(out_layer_2)

# --------------------------------------------------
# model: 2 hidden layers

nhidden_1 = 32;
W_1 = tf.Variable(tf.truncated_normal([vsize, nhidden_1], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[nhidden_1]))
out_layer_1 = tf.nn.relu(tf.matmul(tf_data, W_1) + b_1)

nhidden_2 = 64;
W_2 = tf.Variable(tf.truncated_normal([nhidden_1, nhidden_2], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.1, shape=[nhidden_2]))
out_layer_2 = tf.nn.relu(tf.matmul(out_layer_1, W_2) + b_2)

W_3 = tf.Variable(tf.truncated_normal([nhidden_2, nclass], stddev=0.1))
b_3 = tf.Variable(tf.constant(0.1, shape=[nclass]))
out_layer_3 = tf.matmul(out_layer_2, W_3) + b_3

forward = tf.nn.softmax(out_layer_3)

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