import tensorflow as tf

cnt = tf.Variable(0, name="cnt")
inc = tf.constant(1, name="inc")

add_op = tf.add(cnt, inc)
up_op = tf.assign(cnt, add_op)

with tf.Session() as sess:
    # sessionの先頭で以下のメソッドを用いて変数の初期化が必要
    # tf.variables_initializer()
    # tf.global_variables_initializer()

    sess.run(tf.global_variables_initializer())

    print(sess.run(up_op))
    print(sess.run(up_op))
    print(sess.run(up_op))