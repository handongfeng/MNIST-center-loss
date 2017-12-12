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
        visualize(output, labels, epoch)
        return


class Centers_Print(Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(self.model.get_layer('centerlosslayer').get_weights())


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


###

def visualize_basic(feat, labels, epoch):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.figure()
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.title('epoch = {}'.format(epoch))
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    # plt.axis('off')
    plt.savefig('./images-basic/epoch-{}-val.png'.format(epoch))


def visualize(feat, labels, epoch):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.figure()
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.title('epoch = {}'.format(epoch))
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    # plt.axis('off')
    plt.savefig('./images/epoch-{}-val.png'.format(epoch))
