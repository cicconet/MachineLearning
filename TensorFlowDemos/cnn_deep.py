from scipy import misc
import numpy as np
import tensorflow as tf
import random

# --------------------------------------------------
# setup

ntrain = 1000 # per class
ntest = 100 # per class
nclass = 10 # number of classes
imsize = 28
nchannels = 1
vsize = imsize*imsize # vector size
batchsize = 100

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'MNIST/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'MNIST/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder("float", shape=[None,imsize,imsize,nchannels])
tf_labels = tf.placeholder("float", shape=[None,nclass])

# --------------------------------------------------
# model

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

conv1 = tf.nn.conv2d(tf_data, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, nclass], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[nclass]))

forward = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# --------------------------------------------------
# loss

cross_entropy = -tf.reduce_sum(tf_labels*tf.log(forward))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

evaluation = tf.equal(tf.argmax(forward,1), tf.argmax(tf_labels,1))
accuracy = tf.reduce_mean(tf.cast(evaluation, "float"))

# --------------------------------------------------
# optimization

sess.run(tf.initialize_all_variables())

batch_xs = np.zeros((batchsize,imsize,imsize,nchannels))
batch_ys = np.zeros((batchsize,nclass))

nsamples = ntrain*nclass
for i in range(1000): # 1000
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={tf_data:batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    # if train_accuracy > 0.9:
    #     break
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # dropout only during training

# --------------------------------------------------
# test

print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

sess.close()