from PIL import Image as im
from scipy import misc
import numpy as np
import tensorflow as tf

# --------------------------------------------------
# setup

ntrain = 1000 # per class
ntest = 100 # per class
nclass = 2 # number of classes
vsize = 2 # vector size
batchsize = 10

Train = np.zeros((ntrain*nclass,vsize))
Test = np.zeros((ntest*nclass,vsize))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

Train[0:ntrain,:] = np.random.normal(2.0,0.5,size=[ntrain,2])
Train[ntrain:2*ntrain] = np.random.normal(3.0,0.5,size=[ntrain,2])
Test[0:ntest,:] = np.random.normal(2.0,0.5,size=[ntest,2])
Test[ntest:2*ntest] = np.random.normal(3.0,0.5,size=[ntest,2])

LTrain[0:ntrain,0] = np.ones((ntrain));
LTrain[ntrain:2*ntrain,1] = np.ones((ntrain));
LTest[0:ntest,0] = np.ones((ntest));
LTest[ntest:2*ntest,1] = np.ones((ntest));

tf_data = tf.placeholder(tf.float32, [None, vsize])
tf_labels = tf.placeholder(tf.float32, [None, nclass])

# --------------------------------------------------
# model: 1 hidden layer

nhidden_1 = 32;
W_1 = tf.Variable(tf.truncated_normal([vsize, nhidden_1], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[nhidden_1]))
out_layer_1 = tf.nn.relu(tf.matmul(tf_data, W_1) + b_1)

W_2 = tf.Variable(tf.truncated_normal([nhidden_1, nclass], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.1, shape=[nclass]))
out_layer_2 = tf.sigmoid(tf.matmul(out_layer_1, W_2) + b_2)

forward = tf.nn.softmax(out_layer_2)

# --------------------------------------------------
# model: 3 hidden layers

# nhidden_1 = 48;
# W_1 = tf.Variable(tf.truncated_normal([vsize, nhidden_1], stddev=0.1))
# b_1 = tf.Variable(tf.constant(0.1, shape=[nhidden_1]))
# out_layer_1 = tf.nn.relu(tf.matmul(tf_data, W_1) + b_1)

# nhidden_2 = 48;
# W_2 = tf.Variable(tf.truncated_normal([nhidden_1, nhidden_2], stddev=0.1))
# b_2 = tf.Variable(tf.constant(0.1, shape=[nhidden_2]))
# out_layer_2 = tf.nn.relu(tf.matmul(out_layer_1, W_2) + b_2)

# nhidden_3 = 48;
# W_3 = tf.Variable(tf.truncated_normal([nhidden_2, nhidden_3], stddev=0.1))
# b_3 = tf.Variable(tf.constant(0.1, shape=[nhidden_3]))
# out_layer_3 = tf.nn.relu(tf.matmul(out_layer_2, W_3) + b_3)

# W_4 = tf.Variable(tf.truncated_normal([nhidden_3, nclass], stddev=0.1))
# b_4 = tf.Variable(tf.constant(0.1, shape=[nclass]))
# out_layer_4 = tf.sigmoid(tf.matmul(out_layer_3, W_4) + b_4)

# forward = tf.nn.softmax(out_layer_4)

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

# --------------------------------------------------
# plot

def imshow(I):
    J = im.fromarray(I,'RGB')
    J.show()

w, h = 512, 512
I = np.zeros((h, w, 3), np.uint8)

n = 75
selrows = np.linspace(0,(h-1)/100,n)
selcols = np.linspace(0,(w-1)/100,n)

ntest = n*n
Test = np.zeros((ntest,vsize))
for i in range(0,n):
    for j in range(0,n):
        Test[i*n+j,:] = selrows[i], selcols[j]

testforward = sess.run(tf.argmax(forward,1), feed_dict={tf_data: Test})

for i in range(0,n):
    for j in range(0,n):
        r = np.round(selrows[i]*100).astype(int)
        c = np.round(selcols[j]*100).astype(int)
        if (testforward[i*n+j] == 0):
            I[r,c] = [127,0,0]
        elif (testforward[i*n+j] == 1):
            I[r,c] = [0,127,0]

for i in range(0,2*ntrain):
    r, c = np.round(Train[i,:]*100).astype(int)
    if r >= 0 and r < h and c >= 0 and c < w:
        if LTrain[i,0] == 1 and LTrain[i,1] == 0:
            I[r,c] = [255,0,0]
        elif LTrain[i,0] == 0 and LTrain[i,1] == 1:
            I[r,c] = [0,255,0]

imshow(I)

sess.close()