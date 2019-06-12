import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3 + 0.1*np.random.standard_normal(100)

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(optimizer)

plt.plot(x_data, y_data, 'r.')
plt.plot(x_data, sess.run(y), 'g-')
plt.axis([0, 1, 0, 1])
plt.xlabel('x'), plt.ylabel('y')
plt.legend(['data','fit'])
plt.show()

sess.close()