from scipy import misc
import numpy as np
import tensorflow as tf
import random

ntrain = 1000 # per class
ntest = 100 # per class
nclass = 2 # number of classes
vsize = 784 # vector size
batchsize = 50

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

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, vsize])
y_ = tf.placeholder("float", shape=[None, nclass])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, nclass])
b_fc2 = bias_variable([nclass])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

batch_xs = np.zeros((batchsize,vsize))
batch_ys = np.zeros((batchsize,nclass))
nsamples = ntrain*nclass
for i in range(1000): # was 20000
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:] = Train[perm[j],:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    if train_accuracy > 0.9:
        break
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: Test, y_: LTest, keep_prob: 1.0}))