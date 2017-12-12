from keras.callbacks import Callback
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


class BasicCall(Callback):

    def on_epoch_end(self, epoch, logs={}):
        data = self.validation_data
        labels = np.argmax(data[1], axis=1)
        model = Model(inputs=self.model.input, outputs=self.model.get_layer('side_out').output)
        output = model.predict(data[0])
        visualize_val(output, labels, epoch)
        return

class CenterLossCall(Callback):

    def on_epoch_end(self, epoch, logs={}):
        data = self.validation_data
        labels = np.argmax(data[1], axis=1)
        model = Model(inputs=self.model.input[0], outputs=self.model.get_layer('side_out').output)
        output = model.predict(data[0])
        visualize_val(output, labels, epoch)
        return

class Centers_print(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(self.model.get_layer('center_loss_out').get_weights())


###

def visualize_val(feat, labels, epoch):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.figure()
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.title('epoch = {}'.format(epoch))
    # plt.axis('off')
    plt.savefig('./images/epoch-{}-val.png'.format(epoch))







