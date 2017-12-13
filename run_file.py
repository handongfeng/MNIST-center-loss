import basic_MNIST
import centerLoss_MNIST

basic_MNIST.run()
centerLoss_MNIST.run(0.001)
centerLoss_MNIST.run(0.1)
