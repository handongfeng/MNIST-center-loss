from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras import losses
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import to_categorical

import util
import my_callbacks
import numpy as np

###

initial_learning_rate = 1e-3
batch_size = 64
epochs = 100
lambda_centerloss = 1

###

centerloss_variable = K.variable(value=0.0, name='lambda_cl_variable')


class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, 2),
                                       initializer='uniform',
                                       trainable=False)
        super(CenterLossLayer, self).build(input_shape)

    def call(self, x, mask=None):
        new_centers = self.centers - self.alpha * K.dot(K.transpose(x[1]),
                                                        (K.dot(x[1], self.centers) - x[0])) / K.transpose(
            1e-3 + K.sum(x[1], axis=0, keepdims=True))
        self.add_update((self.centers, new_centers), x)

        t1 = x[0] - K.dot(x[1], self.centers)
        t1 = K.sum(t1 ** 2, axis=1, keepdims=True)
        t2 = K.dot(x[1], self.centers)
        t2 = K.sum(t2 ** 2, axis=1, keepdims=True)
        self.result = t1 / t2
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def zero_loss(y_true, y_pred):
    return K.mean(y_pred, axis=0)


###

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)


###

def my_model(x, labels):
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
    x = Dense(2, name='side_out')(x)
    #
    main = Dense(10, activation='softmax', name='main_out')(x)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([x, labels])
    return main, side


###

main_input = Input((28, 28, 1))
aux_input = Input((10,))

final_output, side_output = my_model(main_input, aux_input)

model = Model(inputs=[main_input, aux_input], outputs=[final_output, side_output])
model.summary()

optim = optimizers.Adam(lr=initial_learning_rate)
model.compile(optimizer=optim,
              loss=[losses.categorical_crossentropy, zero_loss],
              loss_weights=[1, centerloss_variable])

util.build_empty_dir('logs')
util.build_empty_dir('images')
call1 = TensorBoard(log_dir='logs')
call2 = my_callbacks.CenterLossCall()
call3 = my_callbacks.ActivateCenterLoss(centerloss_variable, lambda_centerloss, 2)
call4 = my_callbacks.Centers_Print()

model.fit([x_train, y_train_onehot], [y_train_onehot, np.zeros((x_train.shape[0], 1))], batch_size=batch_size,
          epochs=epochs,
          verbose=1, validation_data=([x_test, y_test_onehot], [y_test_onehot, np.zeros((x_test.shape[0], 1))]),
          callbacks=[call1, call2, call3, call4])
