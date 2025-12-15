import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_vgg_mnist(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Dropout(0.25))
    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Dropout(0.25))
    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.Dropout(0.25))
    # 分类层
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# -------------------------
# 加载并预处理 MNIST 数据
# -------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 将像素值归一化到 [0,1]，并添加通道维度
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# 如果输入是 (28,28)，扩展为 (28,28,1)
x_train = x_train[..., tf.newaxis]
x_test  = x_test[..., tf.newaxis]

# 可选：打乱训练集（fit 的 validation_split 会在内部打乱，但显式打乱也常用）
# idx = np.random.permutation(len(x_train))
# x_train, y_train = x_train[idx], y_train[idx]

# -------------------------
# 构建、编译与训练模型
# -------------------------
vgg_model = build_vgg_mnist(input_shape=(28,28,1), num_classes=10)
vgg_model.summary()

vgg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7)

history_vgg = vgg_model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 评估
test_loss, test_acc = vgg_model.evaluate(x_test, y_test, verbose=2)
print(f'\nVGG测试准确率: {test_acc:.4f}')
