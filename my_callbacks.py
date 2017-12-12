from keras.callbacks import Callback
from keras.models import Model
from keras import backend  as K
import numpy as np
import matplotlib.pyplot as plt


class BasicCall(Callback):

    def on_epoch_end(self, epoch, logs={}):
        data = self.validation_data
        labels = np.argmax(data[1], axis=1)
        model = Model(inputs=self.model.input, outputs=self.model.get_layer('side_out').output)
        output = model.predict(data[0])
        visualize_basic(output, labels, epoch)
        return


class CenterLossCall(Callback):

    def on_epoch_end(self, epoch, logs={}):
        data = self.validation_data
        labels = np.argmax(data[1], axis=1)
        model = Model(inputs=self.model.input[0], outputs=self.model.get_layer('side_out').output)
        output = model.predict(data[0])
        centers = self.model.get_layer('centerlosslayer').get_weights()[0]
        visualize(output, labels, epoch, centers)
        return


class Centers_Print(Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('---')
        print(type(self.model.get_layer('centerlosslayer').get_weights()))
        print(len(self.model.get_layer('centerlosslayer').get_weights()))
        print(self.model.get_layer('centerlosslayer').get_weights()[0])
        print('---')


class ActivateCenterLoss(Callback):

    def __init__(self, variable, value, threshold=1):
        super().__init__()
        self.variable = variable
        self.value = value
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 < self.threshold:
            pass
        else:
            K.set_value(self.variable, self.value)


class Alpha_Print(Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('---')
        print(type(self.model.get_layer('side_out').get_weights()))
        print(len(self.model.get_layer('side_out').get_weights()))
        print(self.model.get_layer('side_out').get_weights()[0])
        print('---')


###

def i2str(i):
    s = str(i)
    if len(s) == 1:
        return '0' + s
    else:
        return s


def visualize_basic(feat, labels, epoch):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.figure()
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.title('epoch = {}'.format(epoch))
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.savefig('./images-basic/epoch-{}-val.png'.format(i2str(epoch)))


def visualize(feat, labels, epoch, centers):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.figure()
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.plot(centers[:, 0], centers[:, 1], 'kx', mew=2, ms=4)
    plt.title('epoch = {}'.format(epoch))
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.savefig('./images/epoch-{}-val.png'.format(i2str(epoch)))
