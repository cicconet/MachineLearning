# TensorFlow...
#   Represents computations as 'graphs' (tensors 'flow' through computation graphs)
# 	Executes graphs in the context of Sessions
# 	Represents data as tensors
# 	Maintains state with Variables
# 	Uses feeds and fetches to get data into and out of arbitrary operations

import tensorflow as tf

matrix = tf.constant([[1., 0.],[0., 1.]])
vector = tf.constant([[1.], [2.]])

prodop = tf.matmul(matrix, vector)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
sumop = tf.add(input1,input2)

sess = tf.Session()
p = sess.run(prodop)
s = sess.run(sumop,feed_dict={input1:p[0],input2:p[1]})
print('product\n',p)
print('sum\n',s)
sess.close()

# with tf.Session() as sess:
# 	result = sess.run(product)
# 	...

# with tf.device("/gpu:0"): ... or /cpu:0, /gpu:1

# ref: https://www.tensorflow.org/versions/r0.9/get_started/basic_usage.html