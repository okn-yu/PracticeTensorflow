import tensorflow as tf

x = tf.constant(1, name="x")
y = tf.placeholder(tf.int32, name="y")

add_op = tf.add(x, y)

#evalメソッドはデータフローグラフ中の1つのノードに対する評価を行う
#runメソッドを用いるとデータフローグラフ全体のノードの演算結果を評価する

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # プレースホルダーには、feed_dictという仕組みを通じて値を外挿できる
    print(sess.run(add_op, feed_dict={y:1}))
    print(sess.run(add_op, feed_dict={y:3}))