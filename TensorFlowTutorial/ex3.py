import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3 + 0.1*np.random.standard_normal(100)

tf_x = tf.placeholder(tf.float32)
tf_y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.zeros([1]))
y = W * tf_x + b

loss = tf.reduce_mean(tf.square(y - tf_y))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

batchsize = 10
batch_x = np.zeros(batchsize)
batch_y = np.zeros(batchsize)
for step in range(201):
	perm = np.arange(100)
	np.random.shuffle(perm)
	batch_x[:] = x_data[perm[0:batchsize]]
	batch_y[:] = y_data[perm[0:batchsize]]
	sess.run(optimizer,feed_dict={tf_x: batch_x, tf_y: batch_y})


plt.plot(x_data, y_data, 'r.')
plt.plot(x_data, sess.run(y,feed_dict={tf_x: x_data}), 'g-')
plt.axis([0, 1, 0, 1])
plt.xlabel('x'), plt.ylabel('y')
plt.legend(['data','fit'])
plt.show()

sess.close()