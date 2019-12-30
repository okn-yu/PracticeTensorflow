import tensorflow as tf

x = tf.constant(value=1, dtype=tf.int32, shape=(), name="x")
#x = tf.constant(1, name="x")
y = tf.constant(value=2, dtype=tf.int32, shape=(), name="y")
#y = tf.constant(2, name="y")

add_op = tf.add(x, y)

with tf.Session() as sess:
    print(sess.run(add_op))