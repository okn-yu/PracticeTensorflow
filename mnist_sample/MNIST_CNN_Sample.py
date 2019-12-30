import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

# 乱数シードを固定
tf.set_random_seed(12345)

# 入力データ
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 入力画像
x = tf.placeholder(tf.float32, name='x')

# サイズ変更
x_1 = tf.reshape(x, [-1, 28, 28, 1])

# 畳み込み
k_0 = tf.Variable(tf.truncated_normal([4, 4, 1, 10], mean=0.0, stddev=0.1))
# 畳み込み
x_2 = tf.nn.conv2d(x_1, k_0, strides=[1, 3, 3, 1], padding='VALID')

# 活性化関数
x_3 = tf.nn.relu(x_2)

# プーリング
x_4 = tf.nn.max_pool(x_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# サイズ変更
x_5 = tf.reshape(x_4, [-1, 160])

# 全結合
w_1 = tf.Variable(tf.zeros([160, 40]))
b_1 = tf.Variable([0.1] * 40)
x_6 = tf.matmul(x_5, w_1) + b_1

# 活性化関数
x_7 = tf.nn.relu(x_6)

# 全結合
w_2 = tf.Variable(tf.zeros([40, 10]))
b_2 = tf.Variable([0.1] * 10)
x_8 = tf.matmul(x_7, w_2) + b_2

# 確率化
y = tf.nn.softmax(x_8)

# 損失関数の最小化
labels = tf.placeholder(tf.float32, name='labels')
# 損失関数と最適化処理（Adam）
loss = -tf.reduce_sum(labels * tf.log(y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 精度検証
prediction_match = tf.equal(tf.argmax(y, axis=1), tf.argmax(labels, axis=1))
accuracy = tf.reduce_mean(tf.cast(prediction_match, tf.float32), name='accuracy')

# パラメーター
BATCH_SIZE = 32
NUM_TRAIN = 10000
OUTPUT_BY = 500

# 学習の実行
sess.run(tf.global_variables_initializer())
for i in range(NUM_TRAIN):
  batch = mnist.train.next_batch(BATCH_SIZE)
  inout = {x: batch[0], labels: batch[1]}
  if i % OUTPUT_BY == 0:
    train_accuracy = accuracy.eval(feed_dict=inout)
    print('step {:d}, accuracy {:.2f}'.format(i, train_accuracy))
  optimizer.run(feed_dict=inout)

# テストデータによる精度検証
test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, labels: mnist.test.labels})
print('test accuracy {:.2f}'.format(test_accuracy))