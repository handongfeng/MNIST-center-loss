from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras import losses
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras import initializers
from keras import backend as K

import utils
import my_callbacks

### parameters

initial_learning_rate = 1e-3
batch_size = 64
epochs = 50
weight_decay = 0.0005


### prelu

def prelu(x, name='default'):
    if name == 'default':
        return PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
    else:
        return PReLU(alpha_initializer=initializers.Constant(value=0.25), name=name)(x)


### model


def basic_model(x):
    x = BatchNormalization()(x)
    #
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Flatten()(x)
    x = Dense(2, kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x, name='side_out')
    return Dense(10, activation='softmax', kernel_regularizer=l2(weight_decay))(x)


###

def run():
    """
    Run the model
    :return:
    """

    ### get data

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)

    ### compile

    inputs = Input((28, 28, 1))
    out = basic_model(inputs)

    model = Model(inputs=inputs, outputs=out)
    model.summary()

    optim = optimizers.SGD(lr=initial_learning_rate, momentum=0.9)
    model.compile(optimizer=optim,
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])

    ### callbacks

    utils.build_empty_dir('logs-basic')
    # utils.build_empty_dir('images-basic')
    call1 = TensorBoard(log_dir='logs-basic')
    call2 = my_callbacks.BasicCall()
    call3 = my_callbacks.Alpha_Print()

    ### fit

    model.fit(x_train, y_train_onehot, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(x_test, y_test_onehot),
              callbacks=[call1, call2, call3])

    ### run training set

    reduced_model = Model(inputs=model.input, outputs=model.get_layer('side_out').output)
    feats = reduced_model.predict(x_train)
    my_callbacks.visualize_basic_train(feats, y_train, epoch=epochs - 1)

    K.clear_session()
    return


###

if __name__ == '__main__':
    run()
