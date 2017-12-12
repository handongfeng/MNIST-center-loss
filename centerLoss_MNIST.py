from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras import losses
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.layers.advanced_activations import PReLU

import util
import my_callbacks
import numpy as np

###

initial_learning_rate = 1e-3
batch_size = 64
epochs = 50
lambda_centerloss = 0.1


###

class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, 2),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] is Nx2, x[1] is Nx10, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        delta_centers /= K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


### custom loss


def zero_loss(y_true, y_pred):
    return K.mean(y_pred, axis=0)


### get data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)


### model

def my_model(x, labels):
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
    #
    main = Dense(10, activation='softmax', name='main_out')(x)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([x, labels])
    return main, side


### compile

main_input = Input((28, 28, 1))
aux_input = Input((10,))

final_output, side_output = my_model(main_input, aux_input)

model = Model(inputs=[main_input, aux_input], outputs=[final_output, side_output])
model.summary()

optim = optimizers.Adam(lr=initial_learning_rate)
model.compile(optimizer=optim,
              loss=[losses.categorical_crossentropy, zero_loss],
              loss_weights=[1, lambda_centerloss])

### callbacks

util.build_empty_dir('logs')
util.build_empty_dir('images')
call1 = TensorBoard(log_dir='logs')
call2 = my_callbacks.CenterLossCall()

### fit

dummy1 = np.zeros((x_train.shape[0], 1))
dummy2 = np.zeros((x_test.shape[0], 1))

model.fit([x_train, y_train_onehot], [y_train_onehot, dummy1], batch_size=batch_size,
          epochs=epochs,
          verbose=2, validation_data=([x_test, y_test_onehot], [y_test_onehot, dummy2]),
          callbacks=[call1, call2])
