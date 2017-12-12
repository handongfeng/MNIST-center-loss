from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras import losses
from keras.utils import to_categorical
from keras.layers.advanced_activations import PReLU

import utils
import my_callbacks

### parameters

initial_learning_rate = 1e-3
batch_size = 64
epochs = 50

### get data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


### model


def basic_model(x):
    x = BatchNormalization()(x)
    #
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Flatten()(x)
    x = Dense(2, name='side_out')(x)
    return Dense(10, activation='softmax')(x)


### compile

inputs = Input((28, 28, 1))
out = basic_model(inputs)

model = Model(inputs=inputs, outputs=out)
model.summary()

optim = optimizers.Adam(lr=initial_learning_rate)
model.compile(optimizer=optim,
              loss=losses.categorical_crossentropy,
              metrics=['accuracy'])

### callbacks

utils.build_empty_dir('logs-basic')
utils.build_empty_dir('images-basic')
call1 = TensorBoard(log_dir='logs-basic')
call2 = my_callbacks.BasicCall()

### fit

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test),
          callbacks=[call1, call2])
