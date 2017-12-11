from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, Embedding, Lambda
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
import keras
from keras import backend as K
import numpy as np
import util
import my_callback

###

initial_learning_rate = 1e-3
batch_size = 64
epochs = 50
lambda_centerloss = 0.1

###

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y_train_onehot = keras.utils.to_categorical(y_train, 10)
y_test_onehot = keras.utils.to_categorical(y_test, 10)


###

def my_model(x):
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Flatten()(x)
    side_out = Dense(2, name='side_out')(x)
    #
    return Dense(10, activation='softmax', name='final_out')(side_out), side_out


###

inputs = Input((28, 28, 1))
final_out, side_out = my_model(inputs)

input_target = Input(shape=(1,))  # single value ground truth labels as inputs
centers = Embedding(10, 2)(input_target)
l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([side_out, centers])

model = Model(inputs=[inputs, input_target], outputs=[final_out, l2_loss])
# model.summary()

optim = optimizers.Adam(lr=initial_learning_rate)
model.compile(optimizer=optim,
              loss=['categorical_crossentropy', lambda y_true, y_pred: y_pred],
              loss_weights=[1, lambda_centerloss], metrics=['accuracy'])

util.build_empty_dir('logs')
call1 = TensorBoard(log_dir='logs')
util.build_empty_dir('images')
call2 = my_callback.SideOutputCenter()

random_y_train = np.random.rand(x_train.shape[0], 1)
random_y_test = np.random.rand(x_test.shape[0], 1)

model.fit([x_train, y_train], [y_train_onehot, random_y_train], batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=([x_test, y_test], [y_test_onehot, random_y_test]),
          callbacks=[call1, call2])

###
