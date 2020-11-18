import tensorflow as tf
import numpy as np

# a = np.array([[1, 2], [3, 4], [5, 6]])
# # b = tf.stack([2, 2 + 1, 2], axis=0)
# # Note the Transpose on every line below
# columns = tf.constant([[0, 1], [0, 1]])
# shape = tf.constant((2, 6))
# c = tf.scatter_nd(columns, a, shape)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(c))
indices = tf.constant([0, 1, 2, 3])
updates = tf.constant([9, 10, 11, 12])
scatter = tf.tile(tf.reshape(tf.range(5),(1,-1)), [2,1])
# shape = tf.constant(tf.shape(indices))
# scatter = tf.scatter_nd(indices, updates, tf.shape(indices))
with tf.Session() as sess:
    print(sess.run(scatter))
    # print(sess.run(tf.shape(scatter)))
