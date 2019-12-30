import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test =  x_test / 255.0

# visualize y_train[0]
plt.imshow(x_train[1], cmap='gray')
plt.colorbar()
plt.show()
print(y_train[0])

model = tf.keras.models.Sequential([
  # 1次元のベクトルに変換
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  # 全結合層
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # ドロップアウト
  tf.keras.layers.Dropout(0.2),
  # 全結合層
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 訓練プロセスを定義
model.compile(optimizer='adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5)

# 訓練プロセスを可視化
plt.plot(history.history['acc'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# テストデータに訓練済みモデルを適用
model.evaluate(x_test, y_test)
