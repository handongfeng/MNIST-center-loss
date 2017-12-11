from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
import util
import my_callback

initial_learning_rate = 1e-3
batch_size = 64
epochs = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()

###

inputs = Input(shape=(28, 28, 1))
#
x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
x = Activation('relu')(x)
x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
#
x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
x = Activation('relu')(x)
x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
#
x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
x = Activation('relu')(x)
x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
#
x = Flatten()(x)
x = Dense(2)(x)
ip1 = Activation('relu')(x)
#
out = Dense(10, activation='softmax')(ip1)

###


model = Model(inputs=[inputs], outputs=[out])
model.summary()

optim = optimizers.Adam(lr=initial_learning_rate)
model.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

util.build_empty_dir('logs')
call1 = TensorBoard(log_dir='logs')
call2 = my_callback.Histories()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          callbacks=[call1, call2])
