import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ntrain = 200 # per class
nclass = 2
batchsize = 10

Train = np.zeros((ntrain*nclass,2))
LTrain = np.zeros((ntrain*nclass,nclass))

Train[0:ntrain/2,:] = np.random.normal(2.0,0.5,size=[ntrain/2,2])
Train[ntrain/2:ntrain,:] = np.random.normal(5.0,0.5,size=[ntrain/2,2])
Train[ntrain:2*ntrain,0] = np.random.normal(4.5,0.5,size=[ntrain])
Train[ntrain:2*ntrain,1] = np.random.normal(3.0,0.5,size=[ntrain])

LTrain[0:ntrain,0] = np.ones((ntrain));
LTrain[ntrain:2*ntrain,1] = np.ones((ntrain));

tf_data = tf.placeholder(tf.float32)
tf_labels = tf.placeholder(tf.float32)

nhidden_1 = 32;
W_1 = tf.Variable(tf.truncated_normal([2, nhidden_1], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[nhidden_1]))
out_layer_1 = tf.nn.relu(tf.matmul(tf_data, W_1) + b_1)

keep_prob = tf.placeholder(tf.float32)
out_layer_1_dropout = tf.nn.dropout(out_layer_1, keep_prob) # <-- dropout setup

W_2 = tf.Variable(tf.truncated_normal([nhidden_1, nclass], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.1, shape=[nclass]))
out_layer_2 = tf.matmul(out_layer_1_dropout, W_2) + b_2 # <-- don't forget to change to 'out_layer_1_dropout' here
forward = tf.nn.softmax(out_layer_2)

cross_entropy = -tf.reduce_sum(tf_labels*tf.log(forward))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

batch_xs = np.zeros((batchsize,2))
batch_ys = np.zeros((batchsize,nclass))
nsamples = ntrain*nclass
for i in range(2000):
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:] = Train[perm[j],:]
        batch_ys[j,:] = LTrain[perm[j],:]
    sess.run(optimizer, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # <-- dropout during training
n = 70
Test = np.zeros((n*n,2))
for i in range(0,n):
    for j in range(0,n):
        Test[i*n+j,:] = np.array([7*i/n, 7*j/n])

testforward = sess.run(tf.argmax(forward,1), feed_dict={tf_data: Test, keep_prob: 1.0}) # <-- no dropout during testing

plt.plot(Test[(testforward == 0),0],Test[(testforward == 0),1],'.r',markersize=2)
plt.plot(Test[(testforward == 1),0],Test[(testforward == 1),1],'.g',markersize=2)
plt.plot(Train[0:ntrain,0],Train[0:ntrain,1],'or',markersize=5)
plt.plot(Train[ntrain:2*ntrain,0],Train[ntrain:2*ntrain,1],'og',markersize=5)
plt.axis('equal')
plt.show()

sess.close()